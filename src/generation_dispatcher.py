import csv
import os
import os.path
import json
import re
from elasticsearch import Elasticsearch
from muppet import DurableChannel, RemoteChannel

__name__ = "generation_dispatcher"

class GenerationDispatcher:
  
  def __init__(self, config, trainingDataset, holdOutDataset, processingStartIndex, processingEndIndex):
    self.config = config
    self.logger = config["logger"]
    self.esClient = Elasticsearch(config["elasticsearch"]["host"] + ":" + str(config["elasticsearch"]["port"]))
    self.trainingDataset = trainingDataset
    self.holdOutDataset = holdOutDataset
    self.config["processingStartIndex"] = processingStartIndex
    self.config["processingEndIndex"] = processingEndIndex
    self.bagOfPhrases = {}
    self.corpusIndex = config["corpus"]["index"]
    self.corpusType = config["corpus"]["type"]
    self.corpusFields = config["corpus"]["text_fields"]
    self.corpusSize = 0
    self.totalPhrasesDispatched = 0
    self.phrasesGenerated = 0
    self.phrasesNotGenerated = 0
    self.timeout = 86400000
    self.dispatcherName = "bayzee.generation.dispatcher"
    if processingEndIndex != None:
      self.dispatcherName += "." + str(processingStartIndex) + "." + str(processingEndIndex)
    self.workerName = "bayzee.generation.worker"
    self.processorIndex = config["processor"]["index"]
    self.processorType = config["processor"]["type"]
    self.processorPhraseType = config["processor"]["type"]+"__phrase"
    self.processingPageSize = config["processingPageSize"]
    config["processor_phrase_type"] = self.processorPhraseType
    
    self.featureNames = map(lambda x: x["name"], config["generator"]["features"])
    for module in config["processor"]["modules"]:
      self.featureNames = self.featureNames + map(lambda x: x["name"], module["features"])

    # creating generation dispatcher
    self.generationDispatcher = DurableChannel(self.dispatcherName, config, self.timeoutCallback)
    
    # creating controle channel
    self.controlChannel = RemoteChannel(self.dispatcherName, config)

  def dispatchToGenerate(self):
    processorIndex = self.config["processor"]["index"]
    phraseProcessorType = self.config["processor"]["type"] + "__phrase"
    nextPhraseIndex = 0
    if self.config["processingStartIndex"] != None: nextPhraseIndex = self.config["processingStartIndex"]
    endPhraseIndex = -1
    if self.config["processingEndIndex"] != None: endPhraseIndex = self.config["processingEndIndex"]

    if endPhraseIndex != -1 and self.processingPageSize > (endPhraseIndex - nextPhraseIndex):
      self.processingPageSize = endPhraseIndex - nextPhraseIndex + 1
    
    while True:
      phrases = self.esClient.search(index=processorIndex, doc_type=phraseProcessorType, body={"from": nextPhraseIndex,"size": self.processingPageSize, "query":{"match_all":{}},"sort":[{"phrase__not_analyzed":{"order":"asc"}}]}, fields=["_id"])
      if len(phrases["hits"]["hits"]) == 0: break
      self.totalPhrasesDispatched += len(phrases["hits"]["hits"])
      floatPrecision = "{0:." + str(self.config["generator"]["floatPrecision"]) + "f}"
      self.logger.info("Generating features from " + str(nextPhraseIndex) + " to " + str(nextPhraseIndex+len(phrases["hits"]["hits"])) + " phrases...")
      for phraseData in phrases["hits"]["hits"]:
        self.logger.info("Dispatching phrase " + phraseData["_id"])
        content = {"phraseId": phraseData["_id"], "type": "generate", "count": 1, "from": self.dispatcherName}
        self.generationDispatcher.send(content, self.workerName, self.timeout)
      nextPhraseIndex += len(phrases["hits"]["hits"])
      if endPhraseIndex != -1 and nextPhraseIndex >= endPhraseIndex: break
    
    while True:
      message = self.generationDispatcher.receive()
      if "phraseId" in message["content"] and message["content"]["phraseId"] > 0:
        self.phrasesGenerated += 1
        self.generationDispatcher.close(message)
        self.logger.info("Generated for " + message["content"]["phraseId"] + str(self.phrasesGenerated) + "/" + str(self.totalPhrasesDispatched))
      
      if (self.phrasesGenerated + self.phrasesNotGenerated) >= self.totalPhrasesDispatched:
        self.controlChannel.send("dying")
        break

    self.__terminate()
    
  def timeoutCallback(self, message):
    config.logger.info("Message timed out: " + str(message))
    if message["content"]["count"] < 5:
      message["content"]["count"] += 1
      self.generationDispatcher.send(message["content"], self.workerName, self.timeout)
    else:
      #log implementation yet to be done for expired phrases
      self.phrasesNotGenerated += 1
      if self.phrasesNotGenerated == self.totalPhrasesDispatched or (self.phrasesGenerated + self.phrasesNotGenerated) == self.totalPhrasesDispatched:
        self.__terminate()

  def __terminate(self):
    self.logger.info(str(self.totalPhrasesDispatched) + " total dispatched")
    self.logger.info(str(self.phrasesGenerated) + " generated")
    self.logger.info(str(self.phrasesNotGenerated) + " failed to generate")
    self.logger.info("Generation complete")
    self.logger.info("Terminating generation dispatcher")

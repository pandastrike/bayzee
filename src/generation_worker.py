import csv
import os
import os.path
import json
import re
import sys
from elasticsearch import Elasticsearch
from muppet import DurableChannel, RemoteChannel

__name__ = "generation_worker"

class GenerationWorker:
  
  def __init__(self, config, trainingDataset, holdOutDataset):
    self.config = config
    self.logger = config["logger"]
    self.esClient = Elasticsearch(config["elasticsearch"]["host"] + ":" + str(config["elasticsearch"]["port"]))
    self.trainingDataset = trainingDataset
    self.holdOutDataset = holdOutDataset
    self.bagOfPhrases = {}
    self.corpusIndex = config["corpus"]["index"]
    self.corpusType = config["corpus"]["type"]
    self.corpusFields = config["corpus"]["text_fields"]
    self.corpusSize = 0
    self.timeout = 6000000
    self.processorIndex = config["processor"]["index"]
    self.processorType = config["processor"]["type"]
    self.processorPhraseType = config["processor"]["type"]+"__phrase"
    count = self.esClient.count(index=self.corpusIndex, doc_type=self.corpusType, body={"query":{"match_all":{}}})
    self.corpusSize = count["count"]
    self.featureNames = map(lambda x: x["name"], config["generator"]["features"])
    for module in config["processor"]["modules"]:
      self.featureNames = self.featureNames + map(lambda x: x["name"], module["features"])
    
    self.workerName = "bayzee.generation.worker"
    self.dispatchers = {}
    
    #creating worker
    self.worker = DurableChannel(self.workerName, config)
  
  def generate(self):
    self.__extractFeatures()

  def __extractFeatures(self):
    while True:
      message = self.worker.receive()
      if message["content"] == "kill":
        message["responseId"] = message["requestId"]
        self.worker.close(message)
        if len(self.dispatchers) == 0:
          self.worker.end()
          break
        else:
          self.worker.send(content="kill", to=self.workerName)
          continue
      elif message["content"]["type"] == "generate":
        if message["content"]["from"] not in self.dispatchers:
          self.dispatchers[message["content"]["from"]] = RemoteChannel(message["content"]["from"], self.config)
          self.dispatchers[message["content"]["from"]].listen(self.unregisterDispatcher)
        phraseId = message["content"]["phraseId"]
        phraseData = self.esClient.get(index=self.processorIndex, doc_type=self.processorPhraseType, id = phraseId)
        floatPrecision = "{0:." + str(self.config["generator"]["floatPrecision"]) + "f}"
        token = phraseData["_source"]["phrase"]
        documentId = phraseData["_source"]["document_id"]
        self.logger.info("Extracted common features for phrase '" + token + "'")
        entry = {}
        shouldMatch = map(lambda x: {"match_phrase":{x:token}}, self.corpusFields)
        query = {"query":{"bool":{"should":shouldMatch}}}
        data = self.esClient.search(index=self.corpusIndex, doc_type=self.corpusType, body=query, explain=True, size=self.corpusSize)
        entry["max_score"] = 0
        maxScore = 0
        avgScore = 0
        maxTermFrequency = 0
        avgTermFrequency = 0
        for hit in data["hits"]["hits"]:
          avgScore += float(hit["_score"])
          numOfScores = 0
          hitTermFrequency = 0
          explanation = json.dumps(hit["_explanation"])
          while len(explanation) > len(token):
            indexOfToken = explanation.find("tf(") + len("tf(")
            if indexOfToken < len("tf("):
              break
            explanation = explanation[indexOfToken:]
            freqToken = explanation.split(")")[0]
            explanation = explanation.split(")")[1]
            if freqToken.find("freq=") >= 0:
              numOfScores += 1
              hitTermFrequency += float(freqToken.split("=")[1])
          if numOfScores > 0 : hitTermFrequency = hitTermFrequency / numOfScores
          if maxTermFrequency < hitTermFrequency: maxTermFrequency = hitTermFrequency 
          avgTermFrequency += hitTermFrequency

        if len(data["hits"]["hits"]) > 0:
          avgTermFrequency = avgTermFrequency * 1.0 / len(data["hits"]["hits"])
        
        if int(data["hits"]["total"]) > 0:
          avgScore = (avgScore * 1.0) / int(data["hits"]["total"])
        
        if data["hits"]["max_score"] != None: 
          maxScore = data["hits"]["max_score"]
        
        if "max_score" in self.featureNames:
          entry["max_score"] = floatPrecision.format(float(maxScore))
        if "doc_count" in self.featureNames:
          entry["doc_count"] = floatPrecision.format(float(data["hits"]["total"]))
        if "avg_score" in self.featureNames:
          entry["avg_score"] = floatPrecision.format(float(avgScore))
        if "max_term_frequency" in self.featureNames:
          entry["max_term_frequency"] = floatPrecision.format(float(maxTermFrequency))
        if "avg_term_frequency" in self.featureNames:
          entry["avg_term_frequency"] = floatPrecision.format(float(avgTermFrequency))
        # get additional features
        for processorInstance in self.config["processor_instances"]:
          processorInstance.extractFeatures(self.config, token, entry)

        phraseData["_source"]["features"] = entry
        if token in self.trainingDataset:
          phraseData["_source"]["is_training"] = self.trainingDataset[token].strip()
        if token in self.holdOutDataset:
          phraseData["_source"]["is_holdout"] = self.holdOutDataset[token].strip()
        self.esClient.index(index=self.processorIndex, doc_type=self.processorPhraseType, id=phraseId, body=phraseData["_source"])
        self.worker.reply(message, {"phraseId": phraseId, "status" : "generated", "type" : "reply"}, 120000000)   
      if message["content"]["type"] == "stop_dispatcher":
        self.worker.reply(message, {"phraseId": -1, "status" : "stop_dispatcher", "type" : "stop_dispatcher"}, self.timeout)        

    self.logger.info("Terminating generation worker")

  def unregisterDispatcher(self, dispatcher, message):
    if message == "dying":
      self.dispatchers.pop(dispatcher, None)

    if len(self.dispatchers) == 0:
      self.worker.send(content="kill", to=self.workerName)

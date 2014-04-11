import csv
import os
import os.path
import json
import re
from elasticsearch import Elasticsearch

__name__ = "generation_dispatcher"

class ClassifierDispatcher:
  
  def __init__(self, config, processingStartIndex, processingEndIndex, processingPageSize):
    self.config = config
    self.esClient = Elasticsearch(config["elasticsearch"]["host"] + ":" + str(config["elasticsearch"]["port"]))
    self.config["processingStartIndex"] = processingStartIndex
    self.config["processingEndIndex"] = processingEndIndex
    self.bagOfPhrases = {}
    
    self.corpusSize = 0
    self.processorIndex = config["processor"]["index"]
    self.processorType = config["processor"]["type"]
    self.processorPhraseType = config["processor"]["type"]+"__phrase"
    self.processingPageSize = processingPageSize
    config["processor_phrase_type"] = self.processorPhraseType
    
    self.featureNames = map(lambda x: x["name"], config["generator"]["features"])
    for module in config["processor"]["modules"]:
      self.featureNames = self.featureNames + map(lambda x: x["name"], module["features"])

  def dispatcher(self, dispatcher):
    processorIndex = self.config["processor"]["index"]
    phraseProcessorType = self.config["processor"]["type"] + "__phrase"
    nextPhraseIndex = 0
    if self.config["processingStartIndex"] != None: nextPhraseIndex = self.config["processingStartIndex"]
    endPhraseIndex = -1
    if self.config["processingEndIndex"] != None: endPhraseIndex = self.config["processingEndIndex"]
    totalPhrases = 0
    print nextPhraseIndex, self.processingPageSize
    while True:
      phrases = self.esClient.search(index=processorIndex, doc_type=phraseProcessorType, body={"from": nextPhraseIndex,"size": self.processingPageSize, "query":{"match_all":{}},"sort":[{"phrase__not_analyzed":{"order":"asc"}}]}, fields=["_id"])
      if len(phrases["hits"]["hits"]) == 0: break
      totalPhrases += len(phrases["hits"]["hits"])
      floatPrecision = "{0:." + str(self.config["generator"]["float_precision"]) + "f}"
      print "Generating features from " + str(nextPhraseIndex) + " to " + str(nextPhraseIndex+len(phrases["hits"]["hits"])) + " phrases..."
      for phraseData in phrases["hits"]["hits"]:
        print "dispatcher sending message for phrase ", phraseData["_id"], "to clasification worker"
        content = {"phraseId": phraseData["_id"], "type": "generate"}
        to = self.config["redis"]["classifier_worker_name"]
        timeout = 60000000
        dispatcher.send(content, to, timeout)
      nextPhraseIndex += len(phrases["hits"]["hits"])
      if endPhraseIndex != -1 and nextPhraseIndex > endPhraseIndex: break
    print "dispatching completed for ", totalPhrases
    count = 0
    while True:
      message = dispatcher.receive()
      count += 1
      if count == totalPhrases:
        break
    print "generation for ", totalPhrases, " completed"
    # dispatcher.close()
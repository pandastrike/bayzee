import sys
import os
import os.path
import re
from elasticsearch import Elasticsearch
from time import sleep

esStopWords = ["a", "an", "and", "are", "as", "at", "be", "but", "by", "for", "if", "in", "into", "is", "it", "no", "not", "of", "on", "or", "such", "that", "the", "their", "then", "there", "these", "they", "this", "to", "was", "will", "with"]

__name__ = "annotation_dispatcher"

class AnnotationDispatcher:
  
  def __init__(self, config, dataDir, processingStartIndex, processingEndIndex, processingPageSize):
    self.config = config
    self.esClient = Elasticsearch(config["elasticsearch"]["host"] + ":" + str(config["elasticsearch"]["port"]))
    self.dataDir = dataDir
    self.bagOfPhrases = {}
    self.corpusIndex = config["corpus"]["index"]
    self.corpusType = config["corpus"]["type"]
    self.corpusFields = config["corpus"]["textFields"]
    self.corpusSize = 0
    self.processorIndex = config["processor"]["index"]
    self.processorType = config["processor"]["type"]
    self.processorPhraseType = config["processor"]["type"] + "__phrase"
    self.processingPageSize = processingPageSize
    self.analyzerIndex = self.corpusIndex + "__analysis__"
    self.config["processingStartIndex"] = processingStartIndex
    self.config["processingEndIndex"] = processingEndIndex
    self.config["processingPageSize"] = processingPageSize
    analyzerIndexSettings = {
      "index":{
        "analysis":{
          "analyzer":{
            "analyzer_shingle":{
              "type": "custom",
              "tokenizer": "standard",
              "filter": ["standard", "lowercase", "filter_shingle"]
            }
          },
          "filter":{
            "filter_shingle":{
              "type": "shingle",
              "max_shingle_size": config["generator"]["max_shingle_size"],
              "min_shingle_size": config["generator"]["min_shingle_size"],
              "output_unigrams": (config["generator"]["min_shingle_size"] == 1)
            },
            "filter_stop":{
              "type": "stop"
            }
          }
        }
      }
    }
    analyzerIndexTypeMapping = {
      "properties":{
        "phrase":{"type":"string"},
        "document_id":{"type":"string", "index": "not_analyzed"},
        "phrase__not_analyzed":{"type":"string","index":"not_analyzed"}
      }
    }

    self.featureNames = map(lambda x: x["name"], config["generator"]["features"])
    for module in config["processor"]["modules"]:
      self.featureNames = self.featureNames + map(lambda x: x["name"], module["features"])

    if processingStartIndex == 0:
      if self.esClient.indices.exists(self.analyzerIndex):
        self.esClient.indices.delete(self.analyzerIndex)
      data = self.esClient.indices.create(self.analyzerIndex, analyzerIndexSettings) 
      self.__deleteOutputFiles
        
    if "annotateFromScratch" not in self.config or self.config["annotateFromScratch"] == True:
      try:
        if self.esClient.indices.exists(self.config["processor"]["index"]):
          self.esClient.indices.delete(self.config["processor"]["index"])
        self.esClient.indices.create(self.config["processor"]["index"])
        self.esClient.indices.put_mapping(index=self.config["processor"]["index"],doc_type=self.processorPhraseType,body=analyzerIndexTypeMapping)
        if self.esClient.indices.exists(self.analyzerIndex):
          self.esClient.indices.delete(self.analyzerIndex)
        data = self.esClient.indices.create(self.analyzerIndex, analyzerIndexSettings) 
      except:
        error = sys.exc_info()
        print "Error occurred during initialization of analyzer index", error
        sys.exit(1)
      else:
        sleep(1)

  def dispatcher(self, dispatcher):
    if "indexPhrases" in self.config and self.config["indexPhrases"] == False: return
    nextDocumentIndex = 0
    if self.config["processingStartIndex"] != None: nextDocumentIndex = self.config["processingStartIndex"]
    endDocumentIndex = -1
    if self.config["processingEndIndex"] != None: endDocumentIndex = self.config["processingEndIndex"]
    docLength = 0
    while True:
      documents = self.esClient.search(index=self.corpusIndex, doc_type=self.corpusType, body={"from": nextDocumentIndex,"size": self.processingPageSize,"query":{"match_all":{}}, "sort":[{"_id":{"order":"asc"}}]}, fields=["_id"])
      if len(documents["hits"]["hits"]) == 0: 
        break
      docLength += len(documents["hits"]["hits"])
      print "dispatching " + str(nextDocumentIndex) + " to " + str(nextDocumentIndex+len(documents["hits"]["hits"])) + " documents..."
      for document in documents["hits"]["hits"]:
        print "dispatcher sending message for document ", document["_id"]
        content = {"_id": document["_id"], "type": "annotate"}
        to = self.config["redis"]["worker_name"]
        timeout = 60000000
        dispatcher.send(content, to, timeout)
      
      nextDocumentIndex += len(documents["hits"]["hits"])
      if endDocumentIndex != -1 and endDocumentIndex <= nextDocumentIndex: 
        break
    print docLength, " total dispatched"
    i = 0
    while True:
      message = dispatcher.receive()
      print message["content"]["documentId"], i
      i += 1
    # dispatcher.close()
    # print "dispatcher closed"

  def __deleteAnalyzerIndex(self):
    if self.esClient.indices.exists(self.analyzerIndex):
        self.esClient.indices.delete(self.analyzerIndex)

  def __deleteOutputFiles(self):
    if os.path.exists(self.dataDir + "/classifier.pickle"):
      os.remove(self.dataDir + "/classifier.pickle")
    if os.path.exists(self.dataDir + "/bad-phrases.csv"):
      os.remove(self.dataDir + "/bad-phrases.csv")
    if os.path.exists(self.dataDir + "/good-phrases.csv"):
      os.remove(self.dataDir + "/good-phrases.csv")
    if os.path.exists(self.dataDir + "/hold-out-set.csv"):
      os.remove(self.dataDir + "/hold-out-set.csv")
    if os.path.exists(self.dataDir + "/test-set.csv"):
      os.remove(self.dataDir + "/test-set.csv")
    if os.path.exists(self.dataDir + "/training-set.csv"):
      os.remove(self.dataDir + "/training-set.csv")

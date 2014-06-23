import sys
import os
import os.path
import re
from elasticsearch import Elasticsearch
from time import sleep
from muppet import DurableChannel, RemoteChannel


esStopWords = ["a", "an", "and", "are", "as", "at", "be", "but", "by", "for", "if", "in", "into", "is", "it", "no", "not", "of", "on", "or", "such", "that", "the", "their", "then", "there", "these", "they", "this", "to", "was", "will", "with"]

__name__ = "annotation_dispatcher"

class AnnotationDispatcher:
  
  def __init__(self, config, processingStartIndex, processingEndIndex):
    self.config = config
    self.logger = config["logger"]
    self.esClient = Elasticsearch(config["elasticsearch"]["host"] + ":" + str(config["elasticsearch"]["port"]))
    self.bagOfPhrases = {}
    self.corpusIndex = config["corpus"]["index"]
    self.corpusType = config["corpus"]["type"]
    self.corpusFields = config["corpus"]["text_fields"]
    self.corpusSize = 0
    self.processorIndex = config["processor"]["index"]
    self.processorType = config["processor"]["type"]
    self.processorPhraseType = config["processor"]["type"] + "__phrase"
    self.processingPageSize = config["processingPageSize"]
    self.analyzerIndex = self.corpusIndex + "__analysis__"
    self.config["processingStartIndex"] = processingStartIndex
    self.config["processingEndIndex"] = processingEndIndex
    self.config["processingPageSize"] = self.processingPageSize
    self.totalDocumentsDispatched = 0
    self.documentsAnnotated = 0
    self.documentsNotAnnotated = 0
    self.lastDispatcher = False
    self.endProcess = False
    self.dispatcherName = "bayzee.annotation.dispatcher"
    self.workerName = "bayzee.annotation.worker"
    self.timeout = 86400000
    if processingEndIndex != None:
      self.dispatcherName += "." + str(processingStartIndex) + "." + str(processingEndIndex)

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
              "max_shingle_size": config["generator"]["maxShingleSize"],
              "min_shingle_size": config["generator"]["minShingleSize"],
              "output_unigrams": (config["generator"]["minShingleSize"] == 1)
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
    corpusSize = self.esClient.count(index=self.corpusIndex, doc_type=self.corpusType, body={"query":{"match_all":{}}})
    self.corpusSize = corpusSize["count"]
    self.featureNames = map(lambda x: x["name"], config["generator"]["features"])
    for module in config["processor"]["modules"]:
      self.featureNames = self.featureNames + map(lambda x: x["name"], module["features"])

    if processingStartIndex == 0:
      if self.esClient.indices.exists(self.analyzerIndex):
        self.esClient.indices.delete(self.analyzerIndex)
      data = self.esClient.indices.create(self.analyzerIndex, analyzerIndexSettings) 
        
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
        self.logger.error("Error occurred during initialization of analyzer index: " + str(error))
        sys.exit(1)
      else:
        sleep(1)

    #dispatcher creation
    self.annotationDispatcher = DurableChannel(self.dispatcherName, config, self.timeoutCallback)

    #remote channel intialisation
    self.controlChannel = RemoteChannel(self.dispatcherName, config)

  def dispatchToAnnotate(self):
    if "indexPhrases" in self.config and self.config["indexPhrases"] == False: return
    nextDocumentIndex = 0
    if self.config["processingStartIndex"] != None: nextDocumentIndex = self.config["processingStartIndex"]
    endDocumentIndex = -1
    if self.config["processingEndIndex"] != None: endDocumentIndex = self.config["processingEndIndex"]
   
    if endDocumentIndex != -1 and self.processingPageSize > (endDocumentIndex - nextDocumentIndex):
      self.processingPageSize = endDocumentIndex - nextDocumentIndex + 1

    self.totalDocumentsDispatched = 0

    while True:
      documents = self.esClient.search(index=self.corpusIndex, doc_type=self.corpusType, body={"from": nextDocumentIndex,"size": self.processingPageSize,"query":{"match_all":{}}, "sort":[{"_id":{"order":"asc"}}]}, fields=["_id"])
      if len(documents["hits"]["hits"]) == 0: 
        break
      self.totalDocumentsDispatched += len(documents["hits"]["hits"])
      self.logger.info("Annotating " + str(nextDocumentIndex) + " to " + str(nextDocumentIndex+len(documents["hits"]["hits"])) + " documents...")
      for document in documents["hits"]["hits"]:
        self.logger.info("Dispatching document " + document["_id"])
        content = {"documentId": document["_id"], "type": "annotate", "count": 1, "from":self.dispatcherName}
        self.annotationDispatcher.send(content, self.workerName)
      nextDocumentIndex += len(documents["hits"]["hits"])
      if endDocumentIndex != -1 and endDocumentIndex <= nextDocumentIndex: 
        break
    
    self.logger.info(str(self.totalDocumentsDispatched) + " documents dispatched")
    while True:
      message = self.annotationDispatcher.receive()
      if "documentId" in message["content"] and message["content"]["documentId"] > 0:
        self.documentsAnnotated += 1
        self.annotationDispatcher.close(message)
        self.logger.info("Annotated document " + message["content"]["documentId"] + " - " + str(self.documentsAnnotated) + "/" + str(self.totalDocumentsDispatched))
      
      if (self.documentsAnnotated + self.documentsNotAnnotated) >= self.totalDocumentsDispatched and not self.lastDispatcher:
        self.controlChannel.send("dying")
        self.annotationDispatcher.end()
        break
    
    self.__terminate()

  def timeoutCallback(self, message):
    if message["content"]["count"] < 5:
      message["content"]["count"] += 1
      self.annotationDispatcher.send(message["content"], self.workerName, self.timeout)
    else:
      #log implementation yet to be done for expired documents
      self.documentsNotAnnotated += 1
      if self.documentsNotAnnotated == self.totalDocumentsDispatched or (self.documentsAnnotated + self.documentsNotAnnotated) == self.totalDocumentsDispatched:
        self.__terminate()

  def __terminate(self):
    self.logger.info(str(self.totalDocumentsDispatched) + " total dispatched")
    self.logger.info(str(self.documentsAnnotated) + " annotated")
    self.logger.info(str(self.documentsNotAnnotated) + " failed to annotate")
    self.logger.info("Annotation complete")
    self.logger.info("Terminating annotation dispatcher")

  def __deleteAnalyzerIndex(self):
    if self.esClient.indices.exists(self.analyzerIndex):
        self.esClient.indices.delete(self.analyzerIndex)

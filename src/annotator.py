import sys
import os
import os.path
import yaml
import json
import re
from elasticsearch import Elasticsearch
from time import sleep

esStopWords = ["a", "an", "and", "are", "as", "at", "be", "but", "by", "for", "if", "in", "into", "is", "it", "no", "not", "of", "on", "or", "such", "that", "the", "their", "then", "there", "these", "they", "this", "to", "was", "will", "with"]

__name__ = "annotator"

class Annotator:
  
  def __init__(self, config, dataDir):
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
    self.analyzerIndex = self.corpusIndex + "__analysis__"
    self.analyzerSettings = {
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

    self.featureNames = map(lambda x: x["name"], config["generator"]["features"])
    for module in config["processor"]["modules"]:
      self.featureNames = self.featureNames + map(lambda x: x["name"], module["features"])

    try:
      if self.esClient.indices.exists(self.config["processor"]["index"]):
        self.esClient.indices.delete(self.config["processor"]["index"])
      self.esClient.indices.create(self.config["processor"]["index"])
      if self.esClient.indices.exists(self.analyzerIndex):
        self.esClient.indices.delete(self.analyzerIndex)
      data = self.esClient.indices.create(self.analyzerIndex, self.analyzerSettings) 
    except:
      error = sys.exc_info()
      print "Error occurred during initialization of analyzer index", error
      sys.exit(1)
    else:
      sleep(1)

  def annotate(self):
    self.__indexPhrases()
    print "Annotating documents and phrases..."
    for processorInstance in self.config["processor_instances"]:
      processorInstance.annotate(self.config)
    self.__deleteAnalyzerIndex()

  def __keify(self, phrase):
    phrase = phrase.strip()
    if len(phrase) == 0:
      return ""
    key = re.sub("[^A-Za-z0-9]", " ", phrase)
    key = " ".join(phrase.split())
    key = key.lower()
    key = "-".join(phrase.split())
    return key

  def __replaceUnderscore(self,shingle):
    token = shingle["token"]
    token = token.replace("_","")
    token = re.sub('\s+', ' ', token).strip()
    shingle["token"] = token
    return shingle
    
  def __filterTokens(self, shingle):
    global esStopWords
    tokens = shingle["token"].split(" ")
    firstToken = tokens[0]
    lastToken = tokens[-1]
    isValid = True
    isValid = (isValid and lastToken != None)
    isValid = (isValid and len(lastToken) > 1)
    isValid = (isValid and not firstToken.replace(".","",1).isdigit())
    isValid = (isValid and not lastToken.replace(".","",1).isdigit())
    isValid = (isValid and firstToken not in esStopWords)
    isValid = (isValid and lastToken not in esStopWords)
    return isValid

  def __indexPhrases(self):
    count = self.esClient.count(index=self.corpusIndex, doc_type=self.corpusType, body={"match_all":{}})
    self.corpusSize = count["count"]
    self.documents = self.esClient.search(index=self.corpusIndex, doc_type=self.corpusType, body={"query":{"match_all":{}}, "size":self.corpusSize}, fields=self.corpusFields)
    print "Generating phrases and their features from " + str(self.corpusSize) + " documents..."
    for document in self.documents["hits"]["hits"]:
      for field in self.corpusFields:
        shingles = []
        if type(document["fields"][field]) is list:
          for element in document["fields"][field]:
            if len(element) > 0:
              shingleTokens = self.esClient.indices.analyze(index=self.analyzerIndex, body=element, analyzer="analyzer_shingle")
              shingles += shingleTokens["tokens"]
        else:
          if len(document["fields"][field]) > 0:
            shingles = self.esClient.indices.analyze(index=self.analyzerIndex, body=document["fields"][field], analyzer="analyzer_shingle")["tokens"]
        shingles = map(self.__replaceUnderscore, shingles)
        shingles = filter(self.__filterTokens, shingles)
        if shingles != None and len(shingles) > 0:
          for shingle in shingles:
            phrase = shingle["token"]
            key = self.__keify(phrase)
            if len(key) > 0:
              if key not in self.bagOfPhrases:
                self.bagOfPhrases[key] = {"phrase": phrase, "document_id": document["_id"]}
      print "Generated from", document["_id"]
    
    for key in self.bagOfPhrases:
      data = self.bagOfPhrases[key]
      self.esClient.index(index=self.processorIndex, doc_type=self.processorPhraseType, id=key, body=data)

  def __deleteAnalyzerIndex(self):
    if self.esClient.indices.exists(self.analyzerIndex):
        self.esClient.indices.delete(self.analyzerIndex)
    if os.path.exists(self.dataDir + "/classifier.pickle"):
      os.remove(self.dataDir + "/classifier.pickle")

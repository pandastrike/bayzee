import sys
import os
import os.path
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
    self.processingPageSize = config["processingPageSize"]
    self.analyzerIndex = self.corpusIndex + "__analysis__"
    
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
        "document_id":{"type":"string"},
        "phrase__not_analyzed":{"type":"string","index":"not_analyzed"}
      }
    }

    self.featureNames = map(lambda x: x["name"], config["generator"]["features"])
    for module in config["processor"]["modules"]:
      self.featureNames = self.featureNames + map(lambda x: x["name"], module["features"])

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
        print "Error occurred during initialization of analyzer index", error
        sys.exit(1)
      else:
        sleep(1)

  def annotate(self):
    print "Annotating documents and phrases..."
    self.__indexPhrases()
    for processorInstance in self.config["processor_instances"]:
      processorInstance.annotate(self.config)
    self.__deleteOutputFiles()

  def __keyify(self, phrase):
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
    if "indexPhrases" in self.config and self.config["indexPhrases"] == False: return
    nextDocumentIndex = 0
    if self.config["processingStartIndex"] != None: nextDocumentIndex = int(self.config["processingStartIndex"])
    endDocumentIndex = -1
    if self.config["processingEndIndex"] != None: endDocumentIndex = int(self.config["processingEndIndex"])
    
    while True:
      documents = self.esClient.search(index=self.corpusIndex, doc_type=self.corpusType, body={"from": nextDocumentIndex,"size": self.processingPageSize,"query":{"match_all":{}}, "sort":[{"_id":{"order":"asc"}}]}, fields=self.corpusFields)
      if len(documents["hits"]["hits"]) == 0: break
      print "Generating shingles from " + str(nextDocumentIndex) + " to " + str(nextDocumentIndex+len(documents["hits"]["hits"])) + " documents..."
      for document in documents["hits"]["hits"]:
        for field in self.corpusFields:
          shingles = []
          if field in document["fields"]:
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
              key = self.__keyify(phrase)
              if len(key) > 0:
                if key not in self.bagOfPhrases:
                  self.bagOfPhrases[key] = {"phrase": phrase,"phrase__not_analyzed": phrase,"document_id": document["_id"]}
      for key in self.bagOfPhrases:
        data = self.bagOfPhrases[key]
        self.esClient.index(index=self.processorIndex, doc_type=self.processorPhraseType, id=key, body=data)
      nextDocumentIndex += len(documents["hits"]["hits"])
      if endDocumentIndex != -1 and endDocumentIndex <= nextDocumentIndex: break

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

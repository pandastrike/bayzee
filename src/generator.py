import csv
import nltk
import math
import orange
import sys
import os
import os.path
import yaml
import json
import re
from nltk.corpus import conll2000
from elasticsearch import Elasticsearch
from time import sleep

esStopWords = ["a", "an", "and", "are", "as", "at", "be", "but", "by", "for", "if", "in", "into", "is", "it", "no", "not", "of", "on", "or", "such", "that", "the", "their", "then", "there", "these", "they", "this", "to", "was", "will", "with"]

__name__ = "generator"

class Generator:
  
  def __init__(self, config, dataDir, trainingDataset, holdOutDataset):
    self.config = config
    self.esClient = Elasticsearch(config["elasticsearch"]["host"] + ":" + str(config["elasticsearch"]["port"]))
    self.dataDir = dataDir
    self.trainingDataset = trainingDataset
    self.holdOutDataset = holdOutDataset
    self.bagOfPhrases = {}
    self.corpusIndex = config["corpus"]["index"]
    self.corpusType = config["corpus"]["type"]
    self.corpusFields = config["corpus"]["textFields"]
    self.corpusSize = 0
    self.processorIndex = config["processor"]["index"]
    self.processorType = config["processor"]["type"]
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
      if self.esClient.indices.exists(self.analyzerIndex):
        self.esClient.indices.delete(self.analyzerIndex)
      data = self.esClient.indices.create(self.analyzerIndex, self.analyzerSettings) 
    except:
      error = sys.exc_info()
      print "Error occurred during initialization of analyzer index", error
      sys.exit(1)
    else:
      sleep(1)

  def generate(self):
    self.__analyzeDocuments()
    self.__writeToFile()
    self.__deleteAnalyzerIndex()

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

  def __analyzeDocuments(self):
    count = self.esClient.count(index=self.corpusIndex, doc_type=self.corpusType, body={"match_all":{}})
    self.corpusSize = count["count"]
    documents = self.esClient.search(index=self.corpusIndex, doc_type=self.corpusType, body={"query":{"match_all":{}}, "size":self.corpusSize}, fields=self.corpusFields)

    print "Generating phrases and their features from " + str(self.corpusSize) + " documents..."
    for document in documents["hits"]["hits"]:
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
          self.__addShinglesToBag(document["_id"], shingles)
      print "Generated from", document["_id"]

  def __addShinglesToBag(self, documentId, shingles):
    floatPrecision = "{0:." + str(self.config["generator"]["float_precision"]) + "f}"
    for shingle in shingles:
      token = shingle["token"]
      if token not in self.bagOfPhrases:
        entry = self.bagOfPhrases[token] = {}
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
        annotatedDocument = self.esClient.get(index=self.processorIndex,doc_type=self.processorType,id=documentId)["_source"]
        for processorInstance in self.config["processor_instances"]:
          processorInstance.addFeatures(self.config, token, entry, annotatedDocument)

  def __writeToFile(self):
    #output files
    holdOutFile = self.dataDir + "/hold-out-set.csv"
    trainingOutFile =  self.dataDir + "/training-set.csv"
    testOutFile = self.dataDir + "/test-set.csv"
    holdOutFile = open(holdOutFile, "w")
    trainingOutFile = open(trainingOutFile, "w")
    testOutFile = open(testOutFile, "w")

    headers = ["m#phrase"] + self.featureNames
    holdOutCSVWriter = csv.writer(holdOutFile)
    trainingOutCSVWriter = csv.writer(trainingOutFile)
    testOutCSVWriter = csv.writer(testOutFile)
    
    #writing headers to output files
    testOutCSVWriter.writerow(headers)
    headers.append("c#is_good")
    trainingOutCSVWriter.writerow(headers)
    holdOutCSVWriter.writerow(headers)

    for phrase in self.bagOfPhrases:
      entry = self.bagOfPhrases[phrase]
      phrase = re.sub("[\,]","",phrase)
      row = [phrase.encode('utf-8')]
      for feature in self.featureNames:
        row.append(entry[feature])
      testOutCSVWriter.writerow(row)
      if phrase in self.trainingDataset:
        row.append(int(self.trainingDataset[phrase]))
        trainingOutCSVWriter.writerow(row)
        row.pop()
      if phrase in self.holdOutDataset:
        row.append(int(self.holdOutDataset[phrase]))
        holdOutCSVWriter.writerow(row)
        row.pop()

  def __deleteAnalyzerIndex(self):
    if self.esClient.indices.exists(self.analyzerIndex):
        self.esClient.indices.delete(self.analyzerIndex)
    if os.path.exists(self.dataDir + "/classifier.pickle"):
      os.remove(self.dataDir + "/classifier.pickle")

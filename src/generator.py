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
  
  def __init__(self, config):
    self.config = config
    self.esClient = Elasticsearch()
    self.documents = None
    self.configFilePath = "config"
    self.bagOfPhrases = {}
    self.data_index = config["data"]["index"]
    self.data_fields = config["data"]["fields"]
    self.hasReference = False
    self.documentsSize = 0
    self.analyzerIndex = self.data_index + "__analysis__"
    self.analyzer_settings = {
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
              "output_unigrams": config["generator"]["output_unigrams"]
            },
            "filter_stop":{
              "type": "stop"
            }
          }
        }
      }
    }
    self.documentType = config["data"]["type"]
    self.stopWords = []
    self.directory = os.path.abspath(os.path.join(__file__,"../.."))
    
    stopWordFile = open(self.directory + "/" + self.configFilePath + "/stop-words.txt")
    for word in stopWordFile.readlines():
      self.stopWords.append(word.strip())
    try:
      #initialize the elasticsearch
      if self.esClient.indices.exists(self.analyzerIndex):
        self.esClient.indices.delete(self.analyzerIndex)
      data = self.esClient.indices.create(self.analyzerIndex, self.analyzer_settings)

      if "reference_index" in config and config["reference"]["index"] != None:
        if "reference_type" in config:
          if self.esClient.indices.exists(index=config["reference"]["index"]):
            if self.esClient.indices.exists_type(index=config["reference"]["index"], doc_type=config["reference"]["type"]):
              self.hasReference = True
              self.reference_index = config["reference"]["index"]
              self.reference_type = config["reference"]["type"]
              self.reference_fields = config["reference"]["fields"]
            else:
              print "type ", config["reference"]["type"], " does not exist for index ", config["reference_index"], " in elasticsearch"
          else:
            print "index ", config["reference"]["index"], " does not exist in elasticsearch"
        else:
          print "type for index ", config["reference"]["index"], " does not exist in config file" 
    except:
      error = sys.exc_info()
      print "Error occurred during initialization of analyzer index", error
    else:
      sleep(1)

  def run(self):
    self.__analyzeDocuments()
    self.__writeToFile()
    self.__deleteIndex()

  def __replaceUnderscore(self,shingle):
    token = shingle["token"]
    token = token.replace("_","")
    token = re.sub('\s+', ' ', token).strip()
    shingle["token"] = token
    return shingle
    
  def __filterTokens(self, shingle):
    tokens = shingle["token"].split(" ")
    firstToken = tokens[0]
    lastToken = tokens[-1]
    isValid = True
    isValid = (isValid and lastToken != None)
    isValid = (isValid and len(lastToken) > 1)
    isValid = (isValid and not firstToken.replace(".","",1).isdigit())
    isValid = (isValid and not lastToken.replace(".","",1).isdigit())
    isValid = (isValid and firstToken not in self.stopWords)
    isValid = (isValid and lastToken not in self.stopWords)
    return isValid

  def __analyzeDocuments(self):
    size = self.esClient.search(index=self.data_index, body={"query":{"match_all":{}}},fields=[])
    self.documentsSize = size["hits"]["total"]
    self.documents = self.esClient.search(index=self.data_index, body={"query":{"match_all":{}},"size":size["hits"]["total"]},fields=self.data_fields)
    print "analyzing ", self.documentsSize, " documents"
    for document in self.documents["hits"]["hits"]:
      print "generating phrases for ", document["_id"]
      for field in self.data_fields:
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


  def __addShinglesToBag(self, documentId, shingles):
    global esStopWords
    fields = self.config["data"]["fields"]
    floatPrecesion = "{0:." + str(self.config["generator"]["float_precesion"]) + "f}"
    features = self.config["generator"]["features"]
    for shingle in shingles:
      token = shingle["token"]
      if token not in self.bagOfPhrases:
        entry = self.bagOfPhrases[token] = {}
        shouldMatch = map(lambda x: {"match_phrase":{x:token}}, fields)
        query = {"query":{"bool":{"should":shouldMatch}}}
        data = self.esClient.search(index=self.data_index, doc_type=self.documentType, body=query, explain=True, size= self.documentsSize)
        entry["max_score"] = 0
        max_score = 0
        avg_score = 0
        max_term_frequency = 0
        avg_term_frequency = 0

        if "max_term_frequency" in features or "avg_term_frequency" in features:
          for hit in data["hits"]["hits"]:
            avg_score += int(hit["_score"])
            numOfScores = 0
            hit_term_frequency = 0
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
                hit_term_frequency += float(freqToken.split("=")[1])
            if numOfScores > 0 : hit_term_frequency = hit_term_frequency / numOfScores
            if max_term_frequency < hit_term_frequency: max_term_frequency = hit_term_frequency 
            avg_term_frequency += hit_term_frequency
        
        if len(data["hits"]["hits"]) > 0:
          avg_term_frequency = avg_term_frequency * 1.0 / len(data["hits"]["hits"])
        
        if int(data["hits"]["total"]) > 0:
          avg_score = (avg_score * 1.0) / int(data["hits"]["total"])
        
        if data["hits"]["max_score"] != None: 
          max_score = data["hits"]["max_score"]
        
        entry["document_id"] = documentId
        if "max_score" in features:
          entry["max_score"] = floatPrecesion.format(float(max_score))
        if "doc_count" in features:
          entry["doc_count"] = floatPrecesion.format(float(data["hits"]["total"]))
        if "avg_score" in features:
          entry["avg_score"] = floatPrecesion.format(float(avg_score))
        if "max_term_frequency" in features:
          entry["max_term_frequency"] = floatPrecesion.format(float(max_term_frequency))
        if "avg_term_frequency" in features:
          entry["avg_term_frequency"] = floatPrecesion.format(float(avg_term_frequency))
        
        if self.hasReference:
          permutations = self.__getPermutationsOfTokens(token.split(" "))
          matchPhrases = map(lambda x: {"match_phrase":{"name": x}}, permutations)
          query = {"bool":{"should":matchPhrases}}
          data = self.esClient.count(index=self.reference_index, doc_type=self.reference_type, body=query)
          if data["count"] > 0:
            entry["phrase_match_" + self.reference_type] = 1
          else:
            entry["phrase_match_" + self.reference_type] = 0
          count = 0
          words = token.split(" ")
          words = filter(lambda x: x not in esStopWords, words)
          tokenLength = len(words)
          if tokenLength > 0:
            for word in token.split(" "):
              query = {"query_string":{"fields":self.reference_fields, "query": word}}
              data = self.esClient.count(index=self.reference_index, doc_type=self.reference_type, body=query)
              count += data["count"] / tokenLength
          entry[self.reference_type+"_similarity"] = count 

  def __getPermutationsOfTokens(self, tokens):
    buf = []
    if len(tokens) == 1:
      return tokens 
    for i,token in enumerate(tokens):
      remaining = []
      remaining += tokens
      remaining.pop(i)
      subPermutations = self.__getPermutationsOfTokens(remaining)
      for sub in subPermutations:
        buf.append([token] + [sub])
    buf = map(lambda x: " ".join(x), buf)
    return buf

  def __writeToFile(self):
    trainingRows = {}
    holdOutRows = {}
    features = self.config["generator"]["features"]
    
    #input files for generation of hold out and training set
    holdInFile = self.directory + "/"  + self.configFilePath + "/" + "hold-out-phrases.csv"
    trainingInFile = self.directory + "/" + self.configFilePath  + "/" + "training-phrases.csv"
    holdInF = open(holdInFile, "r")
    trainingInF = open(trainingInFile, "r")

    #output files
    holdOutFile = self.directory + "/data/hold-out-set.csv"
    trainingOutFile =  self.directory + "/data/training-set.csv"
    testOutFile = self.directory + "/data/test-set.csv"
    holdOutFile = open(holdOutFile, "w")
    trainingOutFile = open(trainingOutFile, "w")
    testOutFile = open(testOutFile, "w")

    #csv writers
    if self.hasReference:
      features.append("phrase_match_" + self.reference_type)
      features.append(self.reference_type+"_similarity")

    headers = ["m#document_id","m#phrase"] + features
    holdOutCSVWriter = csv.writer(holdOutFile)
    trainingOutCSVWriter = csv.writer(trainingOutFile)
    testOutCSVWriter = csv.writer(testOutFile)
    
    #writing headers to output files
    testOutCSVWriter.writerow(headers)
    headers.append("c#is_good")
    trainingOutCSVWriter.writerow(headers)
    holdOutCSVWriter.writerow(headers)

    for row in holdInF.readlines()[1:]:
      holdOutRows[row.split(",")[0]] = row.split(",")[1]

    for row in trainingInF.readlines()[1:]:
      trainingRows[row.split(",")[0]] = row.split(",")[1]

    for phrase in self.bagOfPhrases:
      entry = self.bagOfPhrases[phrase]
      phrase = re.sub("[\,]","",phrase)
      row = [entry["document_id"], phrase]
      row = [x.encode('utf-8') for x in row]
      for feature in features:
        if feature in entry:
          row.append(entry[feature])
        else:
          print phrase, entry["document_id"]
      testOutCSVWriter.writerow(row)
      if phrase in trainingRows:
        row.append(int(trainingRows[phrase]))
        trainingOutCSVWriter.writerow(row)
        row.pop()
      if phrase in holdOutRows:
        row.append(int(holdOutRows[phrase]))
        holdOutCSVWriter.writerow(row)
        row.pop()

  def __deleteIndex(self):
    if self.esClient.indices.exists(self.analyzerIndex):
        self.esClient.indices.delete(self.analyzerIndex)
    if os.path.exists(self.directory + "/data/classifier.pickle"):
      os.remove(self.directory + "/data/classifier.pickle")

import sys
import os
import os.path
import re
from elasticsearch import Elasticsearch
from time import sleep

esStopWords = ["a", "an", "and", "are", "as", "at", "be", "but", "by", "for", "if", "in", "into", "is", "it", "no", "not", "of", "on", "or", "such", "that", "the", "their", "then", "there", "these", "they", "this", "to", "was", "will", "with"]

__name__ = "annotation_worker"

class AnnotationWorker:
  
  def __init__(self, config):
    self.config = config
    self.esClient = Elasticsearch(config["elasticsearch"]["host"] + ":" + str(config["elasticsearch"]["port"]))
    self.bagOfPhrases = {}
    self.corpusIndex = config["corpus"]["index"]
    self.corpusType = config["corpus"]["type"]
    self.corpusFields = config["corpus"]["textFields"]
    self.corpusSize = 0
    self.processorIndex = config["processor"]["index"]
    self.processorType = config["processor"]["type"]
    self.processorPhraseType = config["processor"]["type"] + "__phrase"
    self.analyzerIndex = self.corpusIndex + "__analysis__"
    
    
  
  def annotate(self, worker):
    while True:
      message = worker.receive()
      documentId = message["content"]["_id"]
      document = self.esClient.get(index=self.corpusIndex, doc_type=self.corpusType, id = documentId, fields=self.corpusFields)
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
        if not self.esClient.exists(index=self.processorIndex, doc_type=self.processorPhraseType, id=key):
          self.esClient.index(index=self.processorIndex, doc_type=self.processorPhraseType, id=key, body=data)
      for processorInstance in self.config["processor_instances"]:
        processorInstance.annotate(self.config, documentId)
      worker.reply(message, {"documentId": documentId, "status" : "processed", "type" : "reply"}, 120000000)

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


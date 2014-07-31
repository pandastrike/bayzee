import sys
import os
import os.path
import re
from elasticsearch import Elasticsearch
from time import sleep
from muppet import DurableChannel, RemoteChannel

esStopWords = ["a", "an", "and", "are", "as", "at", "be", "but", "by", "for", "if", "in", "into", "is", "it", "no", "not", "of", "on", "or", "such", "that", "the", "their", "then", "there", "these", "they", "this", "to", "was", "will", "with"]

__name__ = "annotation_worker"

class AnnotationWorker:
  
  def __init__(self, config):
    self.config = config
    self.logger = config["logger"]
    self.esClient = Elasticsearch(config["elasticsearch"]["host"] + ":" + str(config["elasticsearch"]["port"]))
    self.corpusIndex = config["corpus"]["index"]
    self.corpusType = config["corpus"]["type"]
    self.corpusFields = config["corpus"]["text_fields"]
    self.corpusSize = 0
    self.workerName = "bayzee.annotation.worker"
    self.timeout = 6000
    self.processorIndex = config["processor"]["index"]
    self.processorType = config["processor"]["type"]
    self.processorPhraseType = config["processor"]["type"] + "__phrase"
    self.analyzerIndex = self.corpusIndex + "__analysis__"
    self.worker = DurableChannel(self.workerName, config)
    self.dispatchers = {}

  def annotate(self):
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
      elif message["content"]["type"] == "annotate":
        if message["content"]["from"] not in self.dispatchers:
          self.dispatchers[message["content"]["from"]] = RemoteChannel(message["content"]["from"], self.config)
          self.dispatchers[message["content"]["from"]].listen(self.unregisterDispatcher)
        documentId = message["content"]["documentId"]
        document = self.esClient.get(index=self.corpusIndex, doc_type=self.corpusType, id = documentId, fields=self.corpusFields)
        if "fields" in document:  
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
                  data = {"phrase": phrase,"phrase__not_analyzed": phrase,"document_id": document["_id"]}
                  if not self.esClient.exists(index=self.processorIndex, doc_type=self.processorPhraseType, id=key):
                    self.esClient.index(index=self.processorIndex, doc_type=self.processorPhraseType, id=key, body=data)
        sleep(1)
        for processorInstance in self.config["processor_instances"]:
          processorInstance.annotate(self.config, documentId)
        self.worker.reply(message, {"documentId": documentId, "status" : "processed", "type" : "reply"}, self.timeout)

    self.logger.info("Terminating annotation worker")

  def unregisterDispatcher(self, dispatcher, message):
    if message == "dying":
      self.dispatchers.pop(dispatcher, None)

    if len(self.dispatchers) == 0:
      self.worker.send(content="kill", to=self.workerName)

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

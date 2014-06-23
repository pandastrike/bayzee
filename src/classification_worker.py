import csv
import math
import orange
import sys
import os.path
import pickle
import nltk
import time
from elasticsearch import Elasticsearch
from muppet import DurableChannel, RemoteChannel

__name__ = "classification_worker"

class ClassificationWorker:

  def __init__(self, config):
    self.config = config
    self.logger = config["logger"]
    self.esClient = Elasticsearch(config["elasticsearch"]["host"] + ":" + str(config["elasticsearch"]["port"]))
    self.trainD = None
    self.classifier = None
    self.phraseId = None
    self.phraseData = None
    self.processorIndex = config["processor"]["index"]
    self.processorType = config["processor"]["type"]
    self.processorPhraseType = config["processor"]["type"]+"__phrase"
    self.features = self.config["generator"]["features"]
    for module in self.config["processor"]["modules"]:
      self.features = self.features + module["features"]

    self.workerName = "bayzee.classification.worker"
    self.timeout = 600000
    self.dispatchers = {}
    
    #creating worker
    self.worker = DurableChannel(self.workerName, config)

  def classify(self):
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
      elif message["content"]["type"] == "classify":
        if message["content"]["from"] not in self.dispatchers:
          self.dispatchers[message["content"]["from"]] = RemoteChannel(message["content"]["from"], self.config)
          self.dispatchers[message["content"]["from"]].listen(self.unregisterDispatcher)
        self.phraseId = message["content"]["phraseId"]
        if self.classifier == None:
          self.trainD = self.__loadDataFromES("train", None)
          self.trainD = orange.Preprocessor_discretize(self.trainD, method=orange.EntropyDiscretization())
          self.__train()

        self.trainD = self.__loadDataFromES("train", None)
        testD = self.__loadDataFromES("test", self.trainD.domain)
      
        self.trainD = orange.Preprocessor_discretize(self.trainD, method=orange.EntropyDiscretization())
        testD = orange.ExampleTable(self.trainD.domain, testD)

        for row in testD:
          phrase = row.getmetas().values()[0].value
          featureSet = {}
          for i,feature in enumerate(self.features):
            featureSet[feature["name"]] = row[i].value

          prob = self.classifier.prob_classify(featureSet).prob("1")
          classType = self.classifier.classify(featureSet)
          self.phraseData["_source"]["prob"] = prob
          self.phraseData["_source"]["class_type"] = classType
          self.logger.info("Classified '" + phrase + "' as " + classType + " with probability " + str(prob))
          self.esClient.index(index=self.processorIndex, doc_type=self.processorPhraseType, id=self.phraseId, body=self.phraseData["_source"])
          self.worker.reply(message, {"phraseId": self.phraseId, "status" : "classified", "type" : "reply"}, 120000000)   

    self.logger.info("Terminating classification worker")

  def __getOrangeVariableForFeature(self, feature):
    if feature["isNumerical"]: 
      return orange.FloatVariable(feature["name"])
    else:
      return orange.EnumVariable(feature["name"])

  def __loadDataFromES(self, dataType, domain):
    table = None
    if dataType != "train":
      table = orange.ExampleTable(domain)
    else:
      attributes = map(self.__getOrangeVariableForFeature, self.features)
      classAttribute = orange.EnumVariable("is_good", values = ["0", "1"])
      domain = orange.Domain(attributes, classAttribute)
      domain.addmeta(orange.newmetaid(), orange.StringVariable("phrase"))
      table = orange.ExampleTable(domain)
    phrases = []
    if dataType == "train":
      phrasesCount = self.esClient.count(index=self.processorIndex, doc_type=self.processorPhraseType, body={"query":{"terms":{"is_training":["1","0"]}}})
      size = phrasesCount["count"]
      phrases = self.esClient.search(index=self.processorIndex, doc_type=self.processorPhraseType, body={"query":{"terms":{"is_training":["1","0"]}}}, size=size)
      phrases = phrases["hits"]["hits"]
    elif dataType == "holdout":
      phraseCount = self.esClient.count(index=self.processorIndex, doc_type=self.processorPhraseType, body={"query":{"terms":{"is_holdout":["1","0"]}}})
      size = phrasesCount["count"]
      phrases = self.esClient.search(index=self.processorIndex, doc_type=self.processorPhraseType, body={"query":{"terms":{"is_holdout":["1","0"]}}}, size=size)
      phrases = phrases["hits"]["hits"]
    else:
      self.phraseData = self.esClient.get(index=self.processorIndex, doc_type=self.processorPhraseType, id=self.phraseId)
      phrases = [self.phraseData]

    for row in phrases:
      try:
        row = row["_source"]
        featureValues = []
        classType = "?"
        for feature in self.features:
          featureValues.append(row["features"][feature["name"]].encode("ascii"))
        if dataType == "train":
          classType = row["is_training"].encode("ascii", "ignore")
        elif dataType == "holdout":
          classType = row["is_holdout"].encode("ascii")
        example = None
        for i,featureValue in enumerate(featureValues):
          attr = domain.attributes[i]
          if type(attr) is orange.EnumVariable: 
            attr.addValue(featureValue)
        example = orange.Example(domain, (featureValues + [classType]))
        example[domain.getmetas().items()[0][0]] = row["phrase"].encode("ascii")
        table.append(example)
      except:
        self.logger.error("Error classifying phrase '" + row["phrase"] + "'")
    return table

  def __train(self):
    for a in self.trainD.domain.attributes:
      self.logger.info("%s: %s" % (a.name,reduce(lambda x,y: x+', '+y, [i for i in a.values])))
    trainSet = []
    for row in self.trainD:
      phrase = row.getmetas().values()[0].value
      classType = row[-1].value

      featureSet = {}
      for i,feature in enumerate(self.features):
        featureSet[feature["name"]] = row[i].value

      trainSet.append((featureSet, classType))

    self.logger.info("\nTraining Naive Bayes Classifier with " + str(len(trainSet)) + " phrases...")
    self.classifier = nltk.NaiveBayesClassifier.train(trainSet)
    
    self.classifier.show_most_informative_features(50)

  def __calculateMeasures(self):
  
    falsePositives = 0
    falseNegatives = 0
    truePositives = 0
    trueNegatives = 0
    totalPositives = 0
    totalNegatives = 0
    totalHoldOutGoodPhrases = 0
    totalHoldOutBadPhrases = 0

    self.trainD = self.__loadDataFromES("train", None)
    self.holdOutD = self.__loadDataFromES("hold", self.trainD.domain)
    self.trainD = orange.Preprocessor_discretize(self.trainD, method=orange.EntropyDiscretization())
    self.holdOutD = orange.ExampleTable(self.trainD.domain, self.holdOutD)
    
    for row in self.holdOutD:
      actualClassType = row[-1].value
      phrase = row.getmetas().values()[0].value

      featureSet = {}
      for i,feature in enumerate(self.features):
        featureSet[feature["name"]] = row[i].value

      if self.classifier == None:
        classifierFile = open(self.classifierFilePath)
        self.classifier = pickle.load(classifierFile)
        classifierFile.close()  
      prob = self.classifier.prob_classify(featureSet).prob("1")
      classType = self.classifier.classify(featureSet)

      if classType == "1":
        totalPositives += 1
        if classType == actualClassType:
          truePositives += 1
      else:
        totalNegatives += 1
        if classType == actualClassType:
          trueNegatives += 1

      if actualClassType == "1":
        totalHoldOutGoodPhrases += 1
      else:
        totalHoldOutBadPhrases += 1

    precisionOfGood = 100.0 * truePositives/totalPositives
    recallOfGood = 100.0 * truePositives/totalHoldOutGoodPhrases
    fMeasureOfGood = 2.0 * precisionOfGood * recallOfGood / (precisionOfGood + recallOfGood)
    precisionOfBad = 100.0 * trueNegatives/totalNegatives
    recallOfBad = 100.0*trueNegatives/totalHoldOutBadPhrases
    fMeasureOfBad = 2.0 * precisionOfBad * recallOfBad / (precisionOfBad + recallOfBad)
    self.logger.info("\nPrecision of Good: " + str(round(precisionOfGood, 2)) + "%")
    self.logger.info("Recall of Good: " + str(round(recallOfGood, 2)) + "%")
    self.logger.info("Balanced F-measure of Good: " + str(round(fMeasureOfGood, 2)) + "%")
    self.logger.info("Precision of Bad: " + str(round(precisionOfBad, 2)) + "%")
    self.logger.info("Recall of Bad: " + str(round(recallOfBad, 2)) + "%")
    self.logger.info("Balanced F-measure of Bad: " + str(round(fMeasureOfBad, 2)) + "%")

  def unregisterDispatcher(self, dispatcher, message):
    if message == "dying":
      self.dispatchers.pop(dispatcher, None)

    if len(self.dispatchers) == 0:
      self.worker.send(content="kill", to=self.workerName)

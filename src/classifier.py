import csv
import math
import orange
import sys
import os.path
import pickle
import nltk
import time
from elasticsearch import Elasticsearch

__name__ = "classifier"

class Classifier:

  def __init__(self, config, dataDir, testFilePath):
    self.config = config
    self.dataDir = dataDir
    self.classifierFilePath = self.dataDir + "/classifier.pickle"
    self.trainD = None
    self.classifier = None
    self.testFilePath = testFilePath
    self.features = self.config["generator"]["features"]
    for module in self.config["processor"]["modules"]:
      self.features = self.features + module["features"]

  def classify(self):
    
    if not os.path.exists(self.classifierFilePath):
      self.trainD = self.__loadDataFromCSV(self.dataDir + "/training-set.csv", "train", None)
      self.trainD = orange.Preprocessor_discretize(self.trainD, method=orange.EntropyDiscretization())
      self.__train()

    if os.path.exists(self.classifierFilePath) and self.testFilePath != None and os.path.exists(self.testFilePath):
      self.trainD = self.__loadDataFromCSV(self.dataDir + "/training-set.csv", "train", None)
      testD = self.__loadDataFromCSV(self.testFilePath, "test", self.trainD.domain)
      
      self.trainD = orange.Preprocessor_discretize(self.trainD, method=orange.EntropyDiscretization())
      testD = orange.ExampleTable(self.trainD.domain, testD)

      classifierFile = open(self.classifierFilePath)
      self.classifier = pickle.load(classifierFile)
      classifierFile.close()

      print("Classifying " + str(len(testD)) + " phrases with Naive Bayes Classifier...")
      goodData = []
      badData = []
      for row in testD:
        phrase = row.getmetas().values()[0].value
        featureSet = {}
        for i,feature in enumerate(self.features):
          featureSet[feature["name"]] = row[i].value

        prob = self.classifier.prob_classify(featureSet).prob("1")
        classType = self.classifier.classify(featureSet)

        if classType == "1":
          goodData.append(tuple([phrase] + [featureSet[feature["name"]] for feature in self.features] + [prob]))
        else:
          badData.append(tuple([phrase] + [featureSet[feature["name"]] for feature in self.features] + [1-prob]))

      with open(self.outputFile + "/good-phrases.csv","a") as out:
        csvOut = csv.writer(out)
        for row in goodData:
          csvOut.writerow(row)

      with open(self.outputFile + "/bad-phrases.csv","a") as out:
        csvOut = csv.writer(out)
        for row in badData:
          csvOut.writerow(row)

    self.__calculateMeasures()

  def __getOrangeVariableForFeature(self, feature):
    if feature["is_numerical"]: 
      return orange.FloatVariable(feature["name"])
    else:
      return orange.EnumVariable(feature["name"])

  def __loadDataFromCSV(self, filename, dataType, domain):
    table = None
    if dataType != "train":
      table = orange.ExampleTable(domain)

    with open(filename,"r") as csvFile:
      csvReader = csv.reader(csvFile)
      classType = "?"
      if dataType == "train":
        attributes = map(self.__getOrangeVariableForFeature, self.features)
        classAttribute = orange.EnumVariable("is_good", values = ["0", "1"])
        domain = orange.Domain(attributes, classAttribute)
        domain.addmeta(orange.newmetaid(), orange.StringVariable("phrase"))
        table = orange.ExampleTable(domain)

      isHeaderRow = True
      for row in csvReader:
        if isHeaderRow:
          isHeaderRow = False 
          continue
        example = None
        featureValues = []
        if dataType != "test": 
          featureValues = row[1:-1]
          classType = row[-1]
        else: 
          featureValues = row[1:]
        for i,featureValue in enumerate(featureValues):
          attr = domain.attributes[i]
          if type(attr) is orange.EnumVariable: 
            attr.addValue(featureValue)
        example = orange.Example(domain, (featureValues + [classType]))
        example[domain.getmetas().items()[0][0]] = row[0]
        table.append(example)
    return table

  def __train(self):
    for a in self.trainD.domain.attributes:
      print "%s: %s" % (a.name,reduce(lambda x,y: x+', '+y, [i for i in a.values]))
    trainSet = []
    for row in self.trainD:
      phrase = row.getmetas().values()[0].value
      classType = row[-1].value

      featureSet = {}
      for i,feature in enumerate(self.features):
        featureSet[feature["name"]] = row[i].value

      trainSet.append((featureSet, classType))

    print("\nTraining Naive Bayes Classifier with " + str(len(trainSet)) + " phrases...")
    self.classifier = nltk.NaiveBayesClassifier.train(trainSet)
    
    # persisting the classifier using pickle
    classifierFile = open(self.classifierFilePath, "wb")
    pickle.dump(self.classifier, classifierFile)
    classifierFile.close()

    self.classifier.show_most_informative_features(50)

    columns = ["phrase"]
    columns = columns + map(lambda x: x["name"], self.features)
    columns = columns + ["probability"]
    with open(self.dataDir + "/good-phrases.csv","w") as out:
      csvOut = csv.writer(out)
      csvOut.writerow(tuple(columns))

    with open(self.dataDir + "/bad-phrases.csv","w") as out:
      csvOut = csv.writer(out)
      csvOut.writerow(tuple(columns))

  def __calculateMeasures(self):
  
    falsePositives = 0
    falseNegatives = 0
    truePositives = 0
    trueNegatives = 0
    totalPositives = 0
    totalNegatives = 0
    totalHoldOutGoodPhrases = 0
    totalHoldOutBadPhrases = 0

    self.trainD = self.__loadDataFromCSV(self.dataDir + "/training-set.csv", "train", None)
    self.holdOutD = self.__loadDataFromCSV(self.dataDir + "/hold-out-set.csv", "hold", self.trainD.domain)
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
    print("\nPrecision of Good: " + str(round(precisionOfGood, 2)) + "%")
    print("Recall of Good: " + str(round(recallOfGood, 2)) + "%")
    print("Balanced F-measure of Good: " + str(round(fMeasureOfGood, 2)))
    print("Precision of Bad: " + str(round(precisionOfBad, 2)) + "%")
    print("Recall of Bad: " + str(round(recallOfBad, 2)) + "%")
    print("Balanced F-measure of Bad: " + str(round(fMeasureOfBad, 2)))
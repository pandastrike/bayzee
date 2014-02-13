import csv
import math
import orange
import sys
import os.path
import pickle
import nltk
import time
from elasticsearch import Elasticsearch
from lib import processor

__name__ = "classifier"

class Classifier:

  def __init__(self, config):
    self.config = config
    self.esClient = Elasticsearch()
    self.documents = None
    self.configFilePath = "env"
    self.bagOfPhrases = {}
    self.data_index = config["data"]["index"]
    self.data_fields = config["data"]["fields"]
    self.data_type = config["data"]["type"]
    self.features = config["generator"]["features"]
    self.documentPosTags = {}
    self.classifierFilePath = os.path.abspath(os.path.join(__file__, "../..", "data/classifier.pickle")) 
    self.trainD = None
    self.classifier = None
    self.directory = os.path.abspath(os.path.join(__file__, "../.."))
    self.testFilePath = os.path.abspath(os.path.join(__file__, "../..", config["classifier"]["input_file"]))
    config["load_data"] = 1
    self.processor = processor.Processor(config)
    self.features = self.features + self.processor.features
    self.outputFile = os.path.abspath(os.path.join(__file__, "../..", config["classifier"]["output_path"]))
    
  def __loadDataFromCSV(self, filename, dataType, domain):
    table = None
    if dataType != "train":
      table = orange.ExampleTable(domain)

    with open(filename,"r") as csvFile:
      csvReader = csv.reader(csvFile)
      classType = "?"
      isHeaderRow = True    
      for row in csvReader:
        if isHeaderRow:
          if dataType == "train":
            self.features = row[2:-1]
            attributes = [orange.FloatVariable(x) for x in self.features]
            classAttribute = orange.EnumVariable("is_good", values = ["0", "1"])
            domain = orange.Domain(attributes, classAttribute)
            domain.addmeta(orange.newmetaid(), orange.StringVariable("id"))
            domain.addmeta(orange.newmetaid(), orange.StringVariable("phrase"))
            table = orange.ExampleTable(domain)
          isHeaderRow = False
          continue
        example = None
        featureValues = []
        if dataType != "test": 
          featureValues = row[2:-1]
          classType = row[-1]
        else: featureValues = row[2:]
        example = orange.Example(domain, (featureValues + [classType]))
        example[domain.getmetas().items()[1][0]] = row[0]
        example[domain.getmetas().items()[0][0]] = row[1]
        table.append(example)
    return table

  def __trainClassifier(self):
    for a in self.trainD.domain.attributes:
      print "%s: %s" % (a.name,reduce(lambda x,y: x+', '+y, [i for i in a.values]))
    train_set = []
    for row in self.trainD:
      docId = row.getmetas().values()[1].value
      phrase = row.getmetas().values()[0].value
      class_type = row[-1].value

      featureSet = {}
      for index,feature in enumerate(self.features):
        featureSet[feature] = row[index].value

      addlFeatures = self.processor.getFeatures(docId, phrase)
      
     
      for key, value in addlFeatures.iteritems():
        featureSet[key] = value

      train_set.append((featureSet, class_type))

    print("\nTraining Naive Bayes Classifier with " + str(len(train_set)) + " phrases...")
    self.classifier = nltk.NaiveBayesClassifier.train(train_set)
    
    # persisting the classifier using pickle
    classifierFile = open(self.classifierFilePath, "wb")
    pickle.dump(self.classifier, classifierFile)
    classifierFile.close()

    self.classifier.show_most_informative_features(50)

    print "Classifier saved to file"

    with open(self.outputFile + "/good-phrases.csv","w") as out:
      csv_out=csv.writer(out)
      csv_out.writerow(tuple(["phrase"] + self.features + ["pos_tags","first_pos_tag","middle_pos_tag","last_pos_tag","avg_word_length","non_alpha_chars","probability"]))
      

    with open(self.outputFile + "/bad-phrases.csv","w") as out:
      csv_out=csv.writer(out)
      csv_out.writerow(tuple(["phrase"] + self.features + ["pos_tags","first_pos_tag","middle_pos_tag","last_pos_tag","avg_word_length","non_alpha_chars","probability"]))

  def __computeClassifierAccuracyMetrics(self):
  
    falsePositives = 0
    falseNegatives = 0
    truePositives = 0
    trueNegatives = 0
    totalPositives = 0
    totalNegatives = 0
    totalHoldOutGoodPhrases = 0
    totalHoldOutBadPhrases = 0

    self.trainD = self.__loadDataFromCSV(self.directory + "/data/training-set.csv", "train", None)
    self.holdOutD = self.__loadDataFromCSV(self.directory + "/data/hold-out-set.csv", "hold", self.trainD.domain)
    self.trainD = orange.Preprocessor_discretize(self.trainD, method=orange.EntropyDiscretization())
    self.holdOutD = orange.ExampleTable(self.trainD.domain, self.holdOutD)
    
    for row in self.holdOutD:
      actual_class_type = row[-1].value
      docId = row.getmetas().values()[1].value
      phrase = row.getmetas().values()[0].value

      featureSet = {}
      for index,feature in enumerate(self.features):
        featureSet[feature] = row[index].value

      addlFeatures = self.processor.getFeatures(docId, phrase)

      for key, value in addlFeatures.iteritems():
        featureSet[key] = value

      if self.classifier == None:
        classifierFile = open(self.classifierFilePath)
        self.classifier = pickle.load(classifierFile)
        classifierFile.close()  
      prob = self.classifier.prob_classify(featureSet).prob("1")
      class_type = self.classifier.classify(featureSet)

      if class_type == "1":
        totalPositives += 1
        if class_type == actual_class_type:
          truePositives += 1
      else:
        totalNegatives += 1
        if class_type == actual_class_type:
          trueNegatives += 1

      if actual_class_type == "1":
        totalHoldOutGoodPhrases += 1
      else:
        totalHoldOutBadPhrases += 1

    precesionOfGood = 100.0 * truePositives/totalPositives
    recallOfGood = 100.0 * truePositives/totalHoldOutGoodPhrases
    precesionOfBad = 100.0 * trueNegatives/totalNegatives
    recallOfBad = 100.0*trueNegatives/totalHoldOutBadPhrases
    fMeasure = 2
    print("\nPrecision of Good: " + str(round(precesionOfGood, 2)) + "%")
    print("Recall of Good: " + str(round(recallOfGood, 2)) + "%")
    print("Balanced F-measure of Good: " + str(round(fMeasure * (precesionOfGood * recallOfGood) / (precesionOfGood + recallOfGood))))
    print("Precision of Bad: " + str(round(precesionOfBad, 2)) + "%")
    print("Recall of Bad: " + str(round(recallOfBad, 2)) + "%")
    print("Balanced F-measure of Bad: " + str(round(fMeasure * (precesionOfBad * recallOfBad) / (precesionOfBad + recallOfBad))))

  def run(self):
    
    if not os.path.exists(self.classifierFilePath):
      self.trainD = self.__loadDataFromCSV(self.directory + "/data/training-set.csv", "train", None)
      self.trainD = orange.Preprocessor_discretize(self.trainD, method=orange.EntropyDiscretization())
      self.__trainClassifier()

    if os.path.exists(self.classifierFilePath) and self.testFilePath != None and os.path.exists(self.testFilePath
      ):
      self.trainD = self.__loadDataFromCSV(self.directory + "/data/training-set.csv", "train", None)
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
        docId = row.getmetas().values()[1].value
        phrase = row.getmetas().values()[0].value
        featureSet = {}
        for index,feature in enumerate(self.features):
          featureSet[feature] = row[index].value

        addlFeatures = self.processor.getFeatures(docId, phrase)

        for key, value in addlFeatures.iteritems():
          featureSet[key] = value

        prob = self.classifier.prob_classify(featureSet).prob("1")
        class_type = self.classifier.classify(featureSet)

        if class_type == "1":
          goodData.append(tuple([phrase] + [featureSet[feature] for feature in self.features] + [addlFeatures["pos_tags"],addlFeatures["first_pos_tag"],addlFeatures["middle_pos_tag"],addlFeatures["last_pos_tag"],addlFeatures["avg_word_length"],addlFeatures["non_alpha_chars"]] + [prob]))
        else:
          badData.append(tuple([phrase] + [featureSet[feature] for feature in self.features] + [addlFeatures["pos_tags"],addlFeatures["first_pos_tag"],addlFeatures["middle_pos_tag"],addlFeatures["last_pos_tag"],addlFeatures["avg_word_length"],addlFeatures["non_alpha_chars"]] + [1-prob]))

      with open(self.outputFile + "/good-phrases.csv","a") as out:
        csv_out=csv.writer(out)
        for row in goodData:
          csv_out.writerow(row)

      with open(self.outputFile + "/bad-phrases.csv","a") as out:
        csv_out=csv.writer(out)
        for row in badData:
          csv_out.writerow(row)

    if self.config["classifier"]["print_measures"] == True:
      self.__computeClassifierAccuracyMetrics()
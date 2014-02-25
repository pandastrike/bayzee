import sys
import yaml
import os
import imp
from src import classifier,generator,annotator
from elasticsearch import Elasticsearch
import csv
import httplib

__name__ = "bayzee"

def __loadConfig(configFilePath):
  config = None
  if not os.path.exists(configFilePath):
    print "Config file does not exist"
    sys.exit(1)
  try:
    dataStream = open(configFilePath, "r")
    config = yaml.load(dataStream)
  except:
    error = sys.exc_info()
    print "Failed to load configuration from file", error
    sys.exit(1)
  else:
    return config

def __loadProcessors(configFilePath, config):
  processorInstances = []
  for module in config["processor"]["modules"]:
    modulePath = os.path.abspath(os.path.join(os.path.dirname(configFilePath), module["path"]))
    processorInstances.append(imp.load_source(module["name"], modulePath))
  config["processor_instances"] = processorInstances

def __getDataDir(configFilePath, config):
  dataDir = os.path.abspath(os.path.join(os.path.dirname(configFilePath), config["output_path"]))
  if not os.path.exists(dataDir):
    os.makedirs(dataDir)
  return dataDir

def annotate(configFilePath):
  config = __loadConfig(configFilePath)
  __loadProcessors(configFilePath, config)
  dataDir = __getDataDir(configFilePath, config)
  ann = annotator.Annotator(config, dataDir)
  ann.annotate()

def generate(configFilePath):
  config = __loadConfig(configFilePath)
  __loadProcessors(configFilePath, config)

  dataDir = __getDataDir(configFilePath, config)

  trainingFilePath = os.path.abspath(os.path.join(os.path.dirname(configFilePath), config["generator"]["training_phrases_file_path"]))
  holdOutFilePath = os.path.abspath(os.path.join(os.path.dirname(configFilePath), config["generator"]["hold_out_phrases_file_path"]))
  trainingFile = open(trainingFilePath, "r")
  holdOutFile = open(holdOutFilePath, "r")
  
  trainingDataset = {}
  for row in trainingFile.readlines()[1:]:
    values = row.split(",")
    trainingDataset[values[0]] = values[1]

  holdOutDataset = {}
  for row in holdOutFile.readlines()[1:]:
    values = row.split(",")
    holdOutDataset[values[0]] = values[1]

  gen = generator.Generator(config, dataDir, trainingDataset, holdOutDataset)
  gen.generate()

def classify(configFilePath, testFilePath):
  config = __loadConfig(configFilePath)
  dataDir = __getDataDir(configFilePath, config)

  if testFilePath == None:
    testFilePath = dataDir + "/test-set.csv"

  cls = classifier.Classifier(config, dataDir, testFilePath)
  cls.classify()
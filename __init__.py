#!/usr/bin/env python
import sys
import yaml
import os
from src import classifier,generator
from lib import processor

dataDirectory = os.path.abspath(os.path.join(__file__,"..","data"))
if not os.path.exists(dataDirectory):
  os.makedirs(dataDirectory)

def __loadConfig(configFilePath):
  config = None
  if not os.path.exists(configFilePath):
    print "config file does not exist"
    sys.exit(0)
  try:
    dataStream = open(configFilePath, "r")
    config = yaml.load(dataStream)
  except:
    error = sys.exc_info()
    print "Error occured", error
  else:
    return config

def create_classifier(configFilePath, testFilePath):
  config = __loadConfig(configFilePath)
  if config != None:
    if testFilePath != None:
      config["classifier"]["input_file"] = testFilePath
    return classifier.Classifier(config)

def create_generator(configFilePath):
  config = __loadConfig(configFilePath)
  if config != None:
    return generator.Generator(config)


def create_processor(configFilePath):
  config = __loadConfig(configFilePath)
  if config != None:
    return processor.Processor(config)


#!/usr/bin/env python
import sys
import yaml
import os
from src import classifier,generator
from lib import processor

def create_classifier(configFilePath, testFilePath):
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
    if testFilePath != None:
      config["classifier"]["input_file"] = testFilePath
    return classifier.Classifier(config)

def create_generator(configFilePath):
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
    return generator.Generator(config)


def create_processor(configFilePath):
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
    return processor.Processor(config)


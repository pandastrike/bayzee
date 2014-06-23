import sys
import yaml
import os
import imp
import logging
from src import annotation_dispatcher, annotation_worker
from src import generation_dispatcher, generation_worker
from src import classification_dispatcher, classification_worker

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

def __initLogger(configFilePath, config):
  logsDir = os.path.abspath(os.path.join(os.path.dirname(configFilePath), config["logger"]["logsDir"]))
  if not os.path.exists(logsDir):
    os.makedirs(logsDir)

  logger = logging.getLogger("bayzee")
  fh = logging.FileHandler(logsDir + "/bayzee.log")
  ch = logging.StreamHandler()
  formatter = logging.Formatter('%(asctime)s - %(name)s - %(levelname)s - %(message)s')
  logger.setLevel(logging.DEBUG)
  fh.setLevel(logging.DEBUG)
  ch.setLevel(logging.DEBUG)
  fh.setFormatter(formatter)
  ch.setFormatter(formatter)
  logger.addHandler(fh)
  logger.addHandler(ch)
  config["logger"] = logger

  logger = logging.getLogger("elasticsearch")
  fh = logging.FileHandler(logsDir + "/elasticsearch.log")
  formatter = logging.Formatter('%(asctime)s - %(name)s - %(levelname)s - %(message)s')
  logger.setLevel(logging.DEBUG)
  fh.setLevel(logging.DEBUG)
  fh.setFormatter(formatter)
  logger.addHandler(fh)

  logger = logging.getLogger("elasticsearch.trace")
  fh = logging.FileHandler(logsDir + "/elasticsearch.trace")
  formatter = logging.Formatter('%(asctime)s - %(name)s - %(levelname)s - %(message)s')
  logger.setLevel(logging.DEBUG)
  fh.setLevel(logging.DEBUG)
  fh.setFormatter(formatter)
  logger.addHandler(fh)

def dispatchToAnnotate(configFilePath, processingStartIndex, processingEndIndex):
  config = __loadConfig(configFilePath)
  __loadProcessors(configFilePath, config)
  __initLogger(configFilePath, config)

  ann = annotation_dispatcher.AnnotationDispatcher(config, processingStartIndex, processingEndIndex)
  ann.dispatchToAnnotate()

def annotate(configFilePath):
  config = __loadConfig(configFilePath)
  __loadProcessors(configFilePath, config)
  __initLogger(configFilePath, config)

  ann = annotation_worker.AnnotationWorker(config)
  ann.annotate()

def dispatchToGenerate(configFilePath, processingStartIndex, processingEndIndex):
  config = __loadConfig(configFilePath)
  __loadProcessors(configFilePath, config)
  __initLogger(configFilePath, config)

  trainingFilePath = os.path.abspath(os.path.join(os.path.dirname(configFilePath), config["generator"]["trainingPhrasesFilePath"]))
  holdOutFilePath = os.path.abspath(os.path.join(os.path.dirname(configFilePath), config["generator"]["holdOutPhrasesFilePath"]))
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

  gen = generation_dispatcher.GenerationDispatcher(config, trainingDataset, holdOutDataset, processingStartIndex, processingEndIndex)
  gen.dispatchToGenerate()

def generate(configFilePath):
  config = __loadConfig(configFilePath)
  __loadProcessors(configFilePath, config)
  __initLogger(configFilePath, config)

  trainingFilePath = os.path.abspath(os.path.join(os.path.dirname(configFilePath), config["generator"]["trainingPhrasesFilePath"]))
  holdOutFilePath = os.path.abspath(os.path.join(os.path.dirname(configFilePath), config["generator"]["holdOutPhrasesFilePath"]))
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

  gen = generation_worker.GenerationWorker(config, trainingDataset, holdOutDataset)
  gen.generate()

def dispatchToClassify(configFilePath, processingStartIndex, processingEndIndex):
  config = __loadConfig(configFilePath)
  __initLogger(configFilePath, config)

  cls = classification_dispatcher.ClassificationDispatcher(config, processingStartIndex, processingEndIndex)
  cls.dispatchToClassify()

def classify(configFilePath):
  config = __loadConfig(configFilePath)
  __initLogger(configFilePath, config)

  cls = classification_worker.ClassificationWorker(config)
  cls.classify()
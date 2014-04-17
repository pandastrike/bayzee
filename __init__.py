import sys
import yaml
import os
import imp
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

def __getDataDir(configFilePath, config):
  dataDir = os.path.abspath(os.path.join(os.path.dirname(configFilePath), config["output_path"]))
  if not os.path.exists(dataDir):
    os.makedirs(dataDir)
  return dataDir

def dispatchToAnnotate(configFilePath, processingStartIndex, processingEndIndex, processingPageSize):
  config = __loadConfig(configFilePath)
  __loadProcessors(configFilePath, config)
  ann = annotation_dispatcher.AnnotationDispatcher(config, processingStartIndex, processingEndIndex, processingPageSize)
  ann.dispatchToAnnotate()

def annotate(configFilePath):
  config = __loadConfig(configFilePath)
  __loadProcessors(configFilePath, config)
  dataDir = __getDataDir(configFilePath, config)
  ann = annotation_worker.AnnotationWorker(config)
  ann.annotate()

def dispatchToGenerate(configFilePath, processingStartIndex, processingEndIndex, processingPageSize):
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
  print "entered"
  gen = generation_dispatcher.GenerationDispatcher(config, dataDir, trainingDataset, holdOutDataset, processingStartIndex, processingEndIndex, processingPageSize)
  gen.dispatchToGenerate()
  

def generate(configFilePath):
  config = __loadConfig(configFilePath)
  __loadProcessors(configFilePath, config)

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

  gen = generation_worker.GenerationWorker(config, trainingDataset, holdOutDataset)
  gen.generate()


def dispatchToClassify(configFilePath, processingStartIndex, processingEndIndex, processingPageSize):
  config = __loadConfig(configFilePath)
  cls = classification_dispatcher.ClassificationDispatcher(config, processingStartIndex, processingEndIndex, processingPageSize)
  cls.dispatchToClassify()

def classify(configFilePath):
  config = __loadConfig(configFilePath)
  cls = classification_worker.ClassificationWorker(config)
  cls.classify()
import csv
import math
import sys
import os.path
import imp
from elasticsearch import Elasticsearch

__name__ = "processor"

class Processor:

  def __init__(self, config):
    self.config = config
    self.esClient = Elasticsearch()
    self.documents = None
    self.configFilePath = "env"
    self.data_index = config["data"]["index"]
    self.data_fields = config["data"]["fields"]
    self.documentType = config["data"]["type"]
    self.processor_index = config["processors"]["index"]
    self.processor_type = config["processors"]["type"]
    self.features = []
    self.annotations = {}
    properties = {}
    processors = []
    
    for name, module in config["processors"]["modules"].iteritems():
      modulePath = os.path.abspath(os.path.join(os.path.dirname(__file__),"..", module["path"]))
      processors.append({"config":module,"module":imp.load_source(module["name"], modulePath)})
      self.features += module["features"]
      if "es_field_name" in module:
        properties[module["es_field_name"]] = module["properties"]
    mapping = {self.config["processors"]["type"]:{"properties":properties}}
    self.processors = processors
    self.properties = properties

    if "load_data" in config:
      self.__loadContent()
    elif config["processors"]["delete_index_on_start"]:
      self.esClient.indices.delete(index=config["processors"]["index"])

    if not self.esClient.indices.exists(index=self.processor_index):
      self.esClient.indices.create(index=config["processors"]["index"], body=mapping)

  def process(self):
    size = self.esClient.search(index=self.data_index, body={"query":{"match_all":{}}}, fields=[])
    self.documentsSize = size["hits"]["total"]
    self.documents = self.esClient.search(index=self.data_index, body={"query":{"match_all":{}},"size":size["hits"]["total"]},fields = self.data_fields)
    
    print "adding features for ", self.documentsSize, " documents"
    for document in self.documents["hits"]["hits"]:
      data = {}
      docId = document["_id"]
      print "processing ",docId
      for processor in self.processors:
        config = processor["config"]
        if "es_field_name" in config:
          data[config["es_field_name"]] = processor["module"].annotateText(document, self.data_fields)
      self.esClient.index(index=self.processor_index, doc_type=self.processor_type, id=docId, body=data)

  def __loadContent(self):
    if self.esClient.indices.exists(index=self.processor_index):
      size = self.esClient.search(index=self.data_index, body={"query":{"match_all":{}}}, fields=[])
      self.documentsSize = size["hits"]["total"]
      self.documents = self.esClient.search(index=self.data_index, body={"query":{"match_all":{}},"size":size["hits"]["total"]},fields = self.data_fields)
      for document in self.documents["hits"]["hits"]:
        docId = document["_id"]
        data = self.esClient.get(index=self.processor_index, doc_type=self.processor_type, id=docId)
        self.annotations[docId] = data["_source"]
    else:
      print "annotations index does not exist please run annotate command"

  def getFeatures(self, docId, phrase):
    features = {}
    data = self.annotations[docId]
    for processor in self.processors:
      config = processor["config"]
      content = ""
      if "es_field_name" in config:
        content = data[config["es_field_name"]]
      moduleFeatures = config["features"]
      processorFeatures = processor["module"].getFeatures(phrase, moduleFeatures, content)
      for feature in moduleFeatures:
        features[feature] = processorFeatures[feature]
    return features

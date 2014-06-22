import re
import nltk
from nltk.corpus import conll2000
from elasticsearch import Elasticsearch

__name__ = "pos_processor"

def trim(value) :
  return value.strip()

def __keyify(phrase):
  phrase = phrase.strip()
  if len(phrase) == 0:
    return ""
  key = re.sub("[^A-Za-z0-9]", " ", phrase)
  key = " ".join(phrase.split())
  key = key.lower()
  key = "-".join(phrase.split())
  return key
  
def getChunkSequence(tree):
  sequence = ""
  for i in range(0,len(tree)):
    if type(tree[i]) == nltk.tree.Tree:
      sequence += tree[i].node
    else:
      sequence += tree[i][1]
  return sequence

class UnigramChunker(nltk.ChunkParserI):
  def __init__(self, train_sents): 
    train_data = [[(t,c) for w,t,c in nltk.chunk.tree2conlltags(sent)]
                  for sent in train_sents]
    self.tagger = nltk.UnigramTagger(train_data)

  def parse(self, sentence): 
    pos_tags = [pos for (word,pos) in sentence]
    tagged_pos_tags = self.tagger.tag(pos_tags)
    chunktags = [chunktag for (pos, chunktag) in tagged_pos_tags]
    conlltags = [(word, pos, chunktag) for ((word,pos),chunktag)
                 in zip(sentence, chunktags)]
    return nltk.chunk.util.conlltags2tree(conlltags)

train_sents = conll2000.chunked_sents('train.txt')
chunker = UnigramChunker(train_sents)

def annotate(config, documentId):
  if "getPosTags" in config and config["getPosTags"] == False: return
  esClient = Elasticsearch(config["elasticsearch"]["host"] + ":" + str(config["elasticsearch"]["port"]))
  corpusIndex = config["corpus"]["index"]
  corpusType = config["corpus"]["type"]
  corpusFields = config["corpus"]["text_fields"]
  processorIndex = config["processor"]["index"]
  processorType = config["processor"]["type"]
  document = esClient.get(index=corpusIndex, doc_type=corpusType, id = documentId, fields=corpusFields)
  content = ""
  if "fields" in document:
    for field in corpusFields:
      if field in document["fields"]:
        if type(document["fields"][field]) is list:
          for element in document["fields"][field]:
            content += element + ". "
        else:
          content += document["fields"][field] + ". "
      
  annotatedDocument = {}
  sentences = nltk.sent_tokenize(content)
  posTaggedSentences = []
  for sentence in sentences:
    sentence = sentence.strip()
    if len(sentence) > 1:
      sentence = sentence.replace("-", " ")
      sentenceWords = nltk.word_tokenize(sentence.lower())
      sentenceWords = map(lambda x: x.replace(".", ""), sentenceWords)
      posTags = nltk.pos_tag(sentenceWords)
      posTaggedSentences.append(posTags)
  if esClient.exists(index=processorIndex, doc_type=processorType, id=document["_id"]):
    annotatedDocument = esClient.get(index=processorIndex, doc_type=processorType, id=document["_id"])["_source"]
  annotatedDocument["pos_tagged_sentences"] = posTaggedSentences
  esClient.index(index=processorIndex, doc_type=processorType, id=document["_id"], body=annotatedDocument)
  config["logger"].info("pos-processor: Annotated document '" + document["_id"] + "'")

def extractFeatures(config, phrase, phraseFeatures):
  processorIndex = config["processor"]["index"]
  processorType = config["processor"]["type"]
  phraseProcessorType = config["processor"]["type"] + "__phrase"
  esClient = Elasticsearch(config["elasticsearch"]["host"] + ":" + str(config["elasticsearch"]["port"]))
  features = phraseFeatures
  phraseData = esClient.get(index=processorIndex, doc_type=phraseProcessorType, id=__keyify(phrase))["_source"]
  documentId = phraseData["document_id"]
  annotatedDocument = esClient.get(index=processorIndex, doc_type=processorType, id=documentId)["_source"]
  posTaggedSentences = annotatedDocument["pos_tagged_sentences"]
  phrase = phraseData["phrase"]
  phrase = phrase.replace("\"", "")
  phraseWords = nltk.word_tokenize(phrase)
  foundMatch = True
  for sentencePosTags in posTaggedSentences:
    posTagString = firstPosTag = middlePosTag = lastPosTag = "X"
    for i, sentencePosTag in enumerate(sentencePosTags):
      if sentencePosTag[1][0:2] == "PO" or sentencePosTag[0] != phraseWords[0] or (i > len(sentencePosTags) - len(phraseWords)):
        foundMatch = False
        continue
      posTagString = firstPosTag = sentencePosTag[1][0:2]
      for j, phraseWord in enumerate(phraseWords[1:]):
        if sentencePosTags[i+j+1][0] != phraseWord:
          break
        posTag = sentencePosTags[i+j+1][1][0:2]
        if posTag != "PO":
          posTagString += posTag
          if len(phraseWords) > 2 and middlePosTag == "X":
            middlePosTag = posTag
          elif j == len(phraseWords) - 2 and lastPosTag == "X":
            lastPosTag = posTag
      if lastPosTag != "X":
        foundMatch = True
        break
      else:
        foundMatch = False
    if foundMatch:
      break
  if not foundMatch:
    posTagString = "X"

  # average word length as a feature
  totalWordLength = 0
  for word in phraseWords:
    totalWordLength += len(word)
  averageWordlength = round(totalWordLength * 1.0/len(phraseWords),2)

  # non alphabet characters in phrase as a feature
  phraseString = phrase.replace(" ", "")
  nonAlphaChars = 0
  for char in phraseString:
    if char.isalpha() == False and char != "'":
      nonAlphaChars += 1
  
  features["pos_tags"] = posTagString
  features["first_pos_tag"] = firstPosTag
  features["middle_pos_tag"] = middlePosTag
  features["last_pos_tag"] = lastPosTag
  features["avg_word_length"] = str(averageWordlength)
  features["non_alpha_chars"] = str(nonAlphaChars)

  config["logger"].info("pos-processor: Extracted features for '" + phrase + "'")

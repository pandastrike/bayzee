import csv
import nltk
import math
import orange
import sys
import os.path
import imp
from nltk.corpus import conll2000
from elasticsearch import Elasticsearch

__name__ = "pos_processor"

def trim(value) :
  return value.strip()

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

def annotateText(document, fields):
  document = document["fields"]
  content = ""
  for field in fields:
    if type(document[field]) is list:
      for element in document[field]:
        content += element + "."
    else:
      content += document[field] + "."

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
  return posTaggedSentences

def getFeatures(phrase, features, posTaggedSentences):
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
  
  return {"avg_word_length": str(averageWordlength), "pos_tags": posTagString, "first_pos_tag":firstPosTag, "middle_pos_tag": middlePosTag, "last_pos_tag": lastPosTag, "non_alpha_chars": str(nonAlphaChars)}
  
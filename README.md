bayzee
======
bayzee uses machine learning to generate domain relevant key phrases from a corpus of plain text documents.
It uses Naive Bayes classification to predict whether a phrase is relevant to the domain of interest or not.
It takes text content indexed in Elasticsearch as input and extracts features that are used by the classifier to predict.
It needs a training set containing manually labeled phrases ('1' if relevant and '0' if not relevant) in csv format.
It also needs a hold-out set for computing the classifier's accuracy.

Following is a high level description of how bayzee works:

* #### Text Annotation
  Text content is pre-processed and annotated and annotated text is indexed in Elasticsearch.
  Standard processor, provided with bayzee, [pos-processor](./lib/pos-processor.py), tags text from the documents in corpus with parts-of-speech (POS Tags) using [NLTK](http://www.nltk.org).
  These POS tags (tuples of word and part-of-speech) are cached in Elasticsearch, so that bayzee doesn't need to annotate on every run.

* #### Feature Extraction
  Text content is analyzed to generate n-word shingles (phrases) that need to be tested for domain relevance.
  Elasticsearch's analyze API is used to generate shingles from text blocks.
  Features are then extracted from these phrases.
  Standard processor, provided with bayzee, [pos-processor](./lib/pos-processor.py), can extract following features for each phrase:
  
  * doc_count: number of documents in the corpus that contain the phrase atleast once
  * max_term_frequency: maximum number of times the phrase occurs in any document in the corpus
  * avg_term_frequency: average number of times the phrase occurs in the corpus
  * max_score: maximum value of TF-IDF score of the phrase across all the documents in the corpus
  * avg_score: average value of TF-IDF score of the phrase across all the documents in the corpus
  * avg_word_length: average length of words in the phrase
  * pos_tags: a string containing sequence of POS tags of words in the phrase
  * first_pos_tag: part-of-speech of the first word in the phrase
  * middle_pos_tag: part-of-speech of the middle word in the phrase
  * last_pos_tag: part-of-speech of the last word in the phrase
  * non_alpha_chars: number of non-alphabetic characters in the phrase
  
  Numerical features are discretized using 'entropy discretization' method from ['orange' package](http://orange.biolab.si)

* #### Phrase Classification
  A manually labelled training data set containing phrases labeled as relevant to domain ('1') or not relevant to domain ('0') is used to train a Naive Bayes classifier.
  Trained classifier is used to predict the probability of each phrase belonging to either of the two classes ('good' or 'bad').
  A manually labelled hold-out data set containing phrases labelled as relevant to domain ('1') or not relevant to domain ('0') is used to evaluate accuracy of the classifier.
  Six measures are computed to evaluate classifier accuracy:
  
  * 'Precision of Good': (# of phrases correctly classified as 'good' / Total # of phrases classified as 'good')
  * 'Recall of Good': (# of phrases correctly classified as 'good' / Total # of 'good' phrases in hold-out)
  * 'Balanced F-measure of Good': 2 * 'Precision of Good' * 'Recall of Good' / ('Precision of Good' + 'Recall of Good')
  * 'Precision of Bad': (# of phrases correctly classified as 'bad' / Total # of phrases classified as 'bad')
  * 'Recall of Bad': (# of phrases correctly classified as 'bad' / Total # of 'bad' phrases in hold-out)
  * 'Balanced F-measure of Bad': 2 * 'Precision of Bad' * 'Recall of Bad' / ('Precision of Bad' + 'Recall of Bad')

## Configuration
  bayzee is configured using YAML configuration file. Following is an example of the config file (inline comments describe each element in the configuration):
  
```yaml
# Elasticsearch server
elasticsearch: 
  # host where Elasticsearch server is running
  host: "127.0.0.1"
  # port on which Elasticsearch server is listening
  port: 9200

#redis storage
redis:
  host: "127.0.0.1"
  port: 6379

# Corpus to use
corpus:
  # name of the Elasticsearch index where the corpus is stored
  index: "products"
  # name of the Elasticsearch document type where the corpus is stored
  type: "product"
  # list of document fields to generate phrases from
  text_fields: ["description"]

timeoutMonitorFrequency: 3600000

# number of documents to process at a time
processingPageSize: 1000

# indicate whether to start annotating from scratch
annotateFromScratch: True
# indicate whether to generate shingles
indexPhrases: True
# indicate whether to generate postags
getPosTags: True

# Processors (add custom processors to list of modules)
processor:
  # name of the Elasticsearch index where annotated text is stored by the processors
  index: "products__annotated"
  # name of the Elasticsearch document type where annotated text is stored by the processors
  type: "product"
  # list of processor modules
  modules:
      # standard bayzee processor to POS tag english text
      # name of the prcessor
    - name: "pos_processor"
      # path to the python module (relative to the location of this config file)
      path: "../lib/pos-processor.py"
      # features that this processor extracts
      features:
        - name: "pos_tags"
          isNumerical: False
        - name: "first_pos_tag"
          isNumerical: False
        - name: "middle_pos_tag"
          isNumerical: False
        - name: "last_pos_tag"
          isNumerical: False
        - name: "avg_word_length"
          isNumerical: True
        - name: "non_alpha_chars"
          isNumerical: True

# Generation
generator:
  # training set file path (relative to the location of this config file)
  trainingPhrasesFilePath: "training-phrases.csv"
  # hold-out set file path (relative to the location of this config file)
  holdOutPhrasesFilePath: "hold-out-phrases.csv"
  # maximum number of words in generated phrase
  maxShingleSize: 3
  # minimum number of words in generated phrase
  minShingleSize: 2
  # list of features to extract
  features:
    - name: "doc_count"
      isNumerical: True
    - name: "max_term_frequency"
      isNumerical: True
    - name: "avg_term_frequency"
      isNumerical: True
    - name: "max_score"
      isNumerical: True
    - name: "avg_score"
      isNumerical: True
  # precision of numerical features
  floatPrecision: 4

# logger config
logger:
  # directory where log files are written (relative to the location of this config file)
  logsDir: "../logs"
```

## Customization
Although, bayzee's standard processor extracts a predefined set of features from text, it is possible to extend bayzee with a custom 'processor' that extracts custom features specific to the domain. Custom processors can be configured in the 'processors' section of the configuration file. Any custom processor module should implement the following two functions:

        annotate(config, documentId) 
        extractFeatures(config, phrase, phraseFeatures)
        
where:

        'config' is a dictionary object containing configuration elements
        'documentId' is the id of the document that is to be annotated
        'phrase' is the phrase for which the features are to be extracted
        'phraseFeatures' is a dictionary object containing the configured features with feature name as the key

See [pos-processor](./lib/pos-processor.py) for an example processor implementation.

## Setup

* Clone the repo
* Install [NLTK](http://www.nltk.org/install.html)
* Install [orange](http://orange.biolab.si/download)
* Install [muppet](https://pypi.python.org/pypi/muppet) `sudo pip install muppet`
* Make sure Elasticsearch server is running and the corpus of documents are indexed in Elasticsearch
* Make sure Redis server is running

## Run

In order to support distributed classification, bayzee uses a dispatcher-worker pattern. A dispatcher sends units of work to worker processes which could be running on different boxes. There are three stages in the classification process: annotation, generation and classification. At each stage in the process, one dispatcher and one or more workers need to be started.

* First, annotate text

  * Start annotation dispatcher

            bin/dispatcher -a `<path-to-config-file>`

  * Start as many annotation workers are you desire

            bin/worker -a `<path-to-config-file>`

* Next, generate phrases and their features

  * Start generation dispatcher

            bin/dispatcher -g `<path-to-config-file>`

  * Start as many generation workers are you desire

            bin/worker -g `<path-to-config-file>`


* Finally, classify phrases

  * Start classification dispatcher

            bin/dispatcher -c `<path-to-config-file>`

  * Start as many classification workers are you desire

            bin/worker -c `<path-to-config-file>`

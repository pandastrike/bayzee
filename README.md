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
  Standard processor, provided with bayzee ('lib/pos-processor.py'), tags text from the documents in corpus with parts-of-speech (POS Tags) using [NLTK](http://www.nltk.org).
  These POS tags (tuples of word and part-of-speech) are cached in Elasticsearch, so that bayzee doesn't need to annotate on every run.

* #### Feature Extraction
  Text content is analyzed to generate n-word shingles (phrases) that need to be tested for domain relevance.
  Elasticsearch's analyze API is used to generate shingles from text blocks.
  Features are then extracted from these phrases.
  Standard processor, provided with bayzee ('lib/pos-processor.py'), can extract following features for each phrase:
  
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

  # Corpus to use
  corpus:
    # name of the Elasticsearch index where the corpus is stored
    index: "example"
    # name of the Elasticsearch document type where the corpus is stored
    type: "product"
    # list of document fields to generate phrases from
    fields: ["name","category","description"]

  # Processors (add custom processors to list of modules)
  processor:
    # name of the Elasticsearch index where annotated text is stored by the processors
    index: "example__annotated"
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
            is_numerical: False
          - name: "first_pos_tag"
            is_numerical: False
          - name: "middle_pos_tag"
            is_numerical: False
          - name: "last_pos_tag"
            is_numerical: False
          - name: "avg_word_length"
            is_numerical: True
          - name: "non_alpha_chars"
            is_numerical: True

  # Generation
  generator:
    # training set file path (relative to the location of this config file)
    training_phrases_file_path: "training-phrases.csv"
    # hold-out set file path (relative to the location of this config file)
    hold_out_phrases_file_path: "hold-out-phrases.csv"
    # maximum number of words in generated phrase
    max_shingle_size: 3
    # minimum number of words in generated phrase
    min_shingle_size: 2
    # list of document's fields to generate phrases from
    fields: ["name","category","description","manufacturer"]
    # list of features to extract
    features:
      - name: "doc_count"
        is_numerical: True
      - name: "max_term_frequency"
        is_numerical: True
      - name: "avg_term_frequency"
        is_numerical: True
      - name: "max_score"
        is_numerical: True
      - name: "avg_score"
        is_numerical: True
    # precision of numerical features
    float_precision: 4

  # output directory (relative to the location of this config file)
  output_path: "../data"
  ```

## Customization
Although, bayzee's standard processor extracts a predefined set of features from text, it is possible to extend bayzee with a custom 'processor' that extracts custom features specific to the domain. Custom processors can be configured in the 'processors' section of the configuration file.

## Setup

* Clone the repo
* Install [NLTK](http://www.nltk.org/install.html)
* Install [orange](http://orange.biolab.si/download)
* Make sure Elasticsearch server is running and the corpus of documents are indexed in Elasticsearch

## Run

* First, annotate text

        bin/run -a <path-to-config-file>

* Next, generate phrases and their features

        bin/run -g <path-to-config-file>

* Finally, classify phrases

        bin/run -c <path-to-config-file>

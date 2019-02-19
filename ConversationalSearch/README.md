# Conversational Search

## Generate BERT Based Embeddings
To generate our Query Embeddings [we used BERT As A Service](https://github.com/hanxiao/bert-as-service) to generate a unique query embedding for each query in our set. If you want to go ahead and regenerate embeddings(or use it to generate another alternate query source for you model) you can follow what we did below.
~~~
cd Data/BERT
pip install bert-serving-server  # server
pip install bert-serving-client  # client, independent of `bert-serving-server`
wget https://storage.googleapis.com/bert_models/2018_10_18/cased_L-24_H-1024_A-16.zip
unzip cased_L-24_H-1024_A-16
bert-serving-start -model_dir ~/Data/BERT/cased_L-24_H-1024_A-16  -num_worker=4 -device_map 0 #depending on your computer play around with these settings
~~~
In a separate shell
~~~
python3 generateQueryEmbeddingsBERT.py ~/Data/MSMARCO/ConversationalSearch/allQueries.tsv ~/Data/MSMARCO/ConversationalSearch/BERTQueryEmbeddings.tsv
~~~

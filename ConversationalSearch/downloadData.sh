export CLOUD_SDK_REPO="cloud-sdk-$(lsb_release -c -s)"
echo "deb http://packages.cloud.google.com/apt $CLOUD_SDK_REPO main" | sudo tee -a /etc/apt/sources.list.d/google-cloud-sdk.list
curl https://packages.cloud.google.com/apt/doc/apt-key.gpg | sudo apt-key add -
sudo apt-get update && sudo apt-get install google-cloud-sdk
sudo apt-get install google-cloud-sdk-app-engine-java
gcloud init
mkdir data
gsutil -m cp -R gs://natural_questions .
mkdir nq
mv natural_questions/v1.0/dev/* nq/
mv natural_questions/v1.0/train/* nq/
rm -rf natural_questions
wget http://qim.fs.quoracdn.net/quora_duplicate_questions.tsv
wget https://msmarco.blob.core.windows.net/msmarcoranking/collectionandqueries.tar.gz
tar -xzvf collectionandqueries.tar.gz
rm collectionandqueries.tar.gz collection.tsv qrels.*
cd nq
gunzip *

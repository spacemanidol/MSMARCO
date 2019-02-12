curl -fsSL https://download.docker.com/linux/ubuntu/gpg | sudo apt-key add -
sudo add-apt-repository "deb [arch=amd64] https://download.docker.com/linux/ubuntu $(lsb_release -cs) stable"
sudo apt-get update
apt-cache policy docker-ce
sudo apt-get install -y docker-ce python-pip docker-compose
pip install tqdm
sudo usermod -aG docker ${USER}
su - ${USER}
id -nG
sudo usermod -aG docker msmarco
git clone https://github.com/dfcf93/MSMARCOV2.git
git clone https://github.com/HiCAL/HiCAL.git
cd HiCAL
# Checkout the sample dataset
git clone https://github.com/hical/sample-dataset.git
cd sample-dataset
python process.py athome4_sample.tgz
# Create the data directory which is mounted to the docker containers
mkdir ../data
cp athome4_sample.tgz athome4_sample_para.tgz ../data/
cd data
wget https://msmarco.blob.core.windows.net/msmarcoranking/collectionandqueries.tar.gz
tar xzvf collectionandqueries.tar.gz
tar xvzf athome4_sample.tgz
mv athome4_test docs
tar xvzf athome4_sample_para.tgz
mv athome4_test para
cd ../
sudo docker-compose -f HiCAL.yml run cal bash
cd src && make corpus_parser
./corpus_parser  --in /data/athome4_sample.tgz --out /data/athome4_sample.bin --para-in /data/athome4_sample_para.tgz --para-out /data/athome4_para_sample.bin
# Use the modified functions.py
cp sample-dataset/functions.py HiCALWeb/hicalweb/interfaces/DocumentSnippetEngine/functions.py
sudo DOC_BIN=/data/athome4_sample.bin PARA_BIN=/data/athome4_para_sample.bin docker-compose -f HiCAL.yml up -d
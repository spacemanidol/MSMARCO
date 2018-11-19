# Duet Baseline
To help community reproduce our results and get started quickly we are including out DUET baseline present in our leaderboard. As of right now only duet.py works as we are separating the items into a predict and train file

## Requirements
Python 3.5
CUDA 9.0 
Pytorch
h5py
numpy
## Setup
1. Download all the MSMARCO ranking files into your datadir
2. Use the existing normalized idf or generate your own normalized idf values using something like our makeidf.py script or any other system you would like.
3. Modify the duet.py script so that DATA_DIR now includes the following: ngraphs.txt, idf.norm.tsv, triples.train.sample.tsv, top1000.dev.tsv qrel.dev.tsv
4. Confirm your enviorment has everything necessary and run duet.py if the model finishes then raise the epoch count to whatever train period you desire. 



# The Rest isnt done.
### Training
1. Modify the config.yaml file to match your desired paramaters such as training epochs, dropout rate, learning rate,etc.
2. Run the following command. If you do not have CUDA set up the --cuda=True will be ignored. --force_restart is not strictly required but it is used to ignore any existing checkpoints in your exp folder. 
~~~
python3 scripts/train.py  --force_restart --cuda=True
~~~
### Prediction
~~~
python3 scripts/predict.py top1000input prediction.tsv --cuda=True
~~~
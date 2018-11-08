# Duet Baseline
To help community reproduce our results and get started quickly we are including out DUET baseline present in our leaderboard

## Requirements
Python 3.5
CUDA 9.0 
Pytorch
h5py
nltk
## Setup
1. Download all the MSMARCO ranking files
2. Generate your own normalized idf values using something like our makeidf.py script or any other system you would like. 
3. Download and veryify your enviorment matches all requirements. 


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
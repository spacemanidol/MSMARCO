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
3. We suggest you use [Conda](https://conda.io/docs/) to create enviroments. Doing so go ahead and create a new enviroment
````
conda create --name msmarcoranking python=3.6 --file req.txt
conda activate msmarcoranking
````
4. Download all the data. We suggest using the downloaddata.sh script. By using this script to agree to the MSMARCO Terms of Service as specified on the [MSMARCO Website](msmarco.org). 
5. We suggest you subsample the eval and dev top1000 files because regular evaluation can take a really long time(5 hours for full official top1000 on a nvidia gtx 1080ti while a 100 query subsample is ~5 minutes).
````
python subsample.py data/top1000.eval.tsv data/top1000.eval.tiny.tsv 100
python subsample.py data/top1000.dev.tsv data/top1000.eval.dev.tsv 100
````
5. Modify the duet script to train for 1 and set the epoch size to 10 epoch and run the script. If everything works as expected you output should about match what is below. 
````
erasmus@spacemanidol:~$ conda activate msmarcoranking
(msmarcoranking) erasmus@spacemanidol:~$ cd MSMARCOV2/Ranking/Baselines/    
(msmarcoranking) erasmus@spacemanidol:~/MSMARCOV2/Ranking/Baselines$ python subsample.py data/top1000.dev.tsv data/top1000.dev.tiny.tsv 10
(msmarcoranking) erasmus@spacemanidol:~/MSMARCOV2/Ranking/Baselines$ python duet.py 
[Dec 12, 20:30:57] Starting
[Dec 12, 20:30:57] No Previous Model Found. Creating new
[Dec 12, 20:30:59] Number of learnable parameters: 39020545
[Dec 12, 20:30:59] Training for 1 epochs
[Dec 12, 20:31:15] epoch:1, loss: 0.6910152077674866
[Dec 12, 20:31:48] MRR @1000:0.004489478973802316
[Dec 12, 20:31:48] Done Training
[Dec 12, 20:31:48] Evaluating and Predicting
[Dec 12, 20:32:20] #####################
[Dec 12, 20:32:20] MRR @10: 0.0
[Dec 12, 20:32:20] QueriesRanked: 10
[Dec 12, 20:32:20] #####################
````
6. Start tinkering. Modify the MB_SIZE depending on the CPU size. 128 seems to work well for the machine I tested it on (bigger caused crashes) with a 1080ti.
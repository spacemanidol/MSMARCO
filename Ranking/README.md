# OneML MARCO hackathon

### Authors:
* Rutger van Haasteren <ruvanh@microsoft.com>
* Daniel Campos <Campos.Daniel@microsoft.com>
* Bhaskar Mitra <Bhaskar.Mitra@microsoft.com>
* Payal Bajaj <Payal.Bajaj@microsoft.com>

### Purpose:
Baseline model and evaluation code for MSMARCO Ranking Dataset

##### Original data files:
Format for the script is found [here](http://faculty.washington.edu/levow/courses/ling573_SPR2011/hw/trec_eval_desc.htm)

The results file has the format: query_id, iter, docno, rank, sim, run_id  delimited by spaces.  
Query id is the query number 
The iter constant, 0, is required but ignored by trec_eval.  
The Document numbers are string values.  
The Similarity (sim) is a float value.  
Rank is an integer from 0 to 1000 
Runid is a string which gets printed out with the output.
 
The evaluation script can be found at [github](https://github.com/usnistgov/trec_eval).

### Usage
```
python msmarco_eval_ranking.py .\eval_ranking.tsv candidate_file.tsv
```

# Online Scoring

# Ranking Task
MS MARCO(Microsoft Machine Reading Comprehension) is a large scale dataset focused on machine reading comprehension, question answering, and passage ranking. 

### Generation
To generate the ranking task dataset we started with the regular MSMARCO dataset which means if people want to generate any data in a different format they are more than able to(and even provide us with suggestions!). We collected all unique passages(without any normalization) to make a pool of ~8.8 million unique passages. Then, for each query from the existing MSMARCO we ran a standard [BM25](https://en.wikipedia.org/wiki/Okapi_BM25) to produce 1000 relevant passages. These were ordered by random so each query now has 1000 corresponding passages. This was done for the train, dev, and evaluation sections of the v2.1 dataset. Understanding that this ranking data may not be useful to train a Deep Learning style system we build the triples.train.tsv files. These 

Then, the existing dataset has an annotation of is_selected:1 if a judge used a passage to generate their answer. We consider these as a ranking signla where all passages that have a value of 1 are a true possitive for query relevance for that given passage. Any passage that has a value of 0 is not a true negative. 
### Data Format
in order to o
## Utilities, Stats and Related Content

### Evaluation
Evaluation of systems will be done using the industry standard BLEU and ROUGE-L. These are far from perfect but have been the best option we have found that scales. If you know of a better metric or want to brainstorm creating one please contact us.

We have made the official evaluation script along with a sample output file on the dev set available for download as well so that you can evaluate your models. [Download the evaluation scripts](https://github.com/dfcf93/MSMARCOV2/tree/master/Q%2BA/Evaluation) The evaluation script takes as inputs a reference and candidate output file. You can execute the evaluation script to evaluate your models as follows:
./run.sh <path to reference json file> <path to candidate json file> 
	
### Leaderboard Results
To Help Teams iterate we are making the results of official submissions on our evaluation script(the scores, not the full submissions)availible. We will update these files as we update metrics and as new submisions come in. They can be found in the [Leaderboard Results](https://github.com/dfcf93/MSMARCOV2/tree/master/Leaderboard%20Results) folder.

### Submissions
Once you have built a model that meets your expectations on evaluation with the dev set, you can submit your test results to get official evaluation on the test set. To ensure the integrity of the official test results, we do not release the correct answers for test set to the public. To submit your model for official evaluation on the test set, follow the below steps:
Run the evaluation script on the test set and generate the output results file for submission
Submit the following information by [contacting us](mailto:ms-marco@microsoft.com?subject=MS Marco Submission)
Individual/Team Name: Name of the individual or the team to appear in the leaderboard [Required]
Individual/Team Institution: Name of the institution of the individual or the team to appear in the leaderboard [Optional]
Model information: Name of the model/technique to appear in the leaderboard [Required]
Paper Information: Name, Citation, URL of the paper if model is from a published work to appear in the leaderboard [Optional]

Please submit your results either in json or jsonl format and ensure that each answer you are providing has its refrence query_id and query_text. If your model does not have query_id and query_text it is difficult/impossible to evalutate the submission.
To avoid "P-hacking" we discourage too many submissions from the same group in a short period of time. Because submissions don't require the final trained model we also retain the right to request a model to validate the results being submitted
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

##### Passage Reranking task Task
Given a query q and a the 1000 most relevant passages P = p1, p2, p3,... p1000, as retrieved by BM25 a succeful system is expected to rerank the most relevant passage as high as possible. For this task not all 1000 relevant items have a human labeled relevant passage. Evaluation will be done using [MRR](https://en.wikipedia.org/wiki/Mean_reciprocal_rank)

# Online Scoring

# MSMARCO V2

MS MARCO(Microsoft Machine Reading Comprehension) is a large scale dataset focused on machine reading comprehension and question answering. In MS MARCO, all question have been generated from real anonymized Bing user queries which grounds the dataset in a real world problem and can provide researchers real contrainsts their models might be used in.The context passages, from which the answers in the dataset are derived, are extracted from real web documents using the most advanced version of the Bing search engine. The answers to the queries are human generated.

First released at [NIPS 2016](https://arxiv.org/pdf/1611.09268v2.pdf) MSMARCO was 100,000 queries with a variety of domains. Since then, the MSMARCO team has been hard at work making the data bigger and better. Details about all the changes/improvements are below. If you have suggestions for our dataset, novel uses, or general feedback please reach out to us at ms-marco@microsoft.com

In its current form(V2.1), there are 1,010,916 unique real queries that were generated by sampling and anonymizing Bing usage logs. After sampling, we used Bing to extract the 10 most relevant passages and asked a human judge to answer the query given the information present. Not all queries had answers present in the passages but we thought this data would be useful to help systems learn that not all answers can be answers.  Around 35% of all queries in the dataset have the answer 'No answer Present' meaning judges were unable to answer the question given the information provided by the 10 relevant passages.


## Dataset Generation, Data Format, And Statistics

#### Generation
The MSMARCO dataset is generated by a well oiled pipeline optimized for the highest quality examples. the general process runs as follows.
1. Bing logs are sampled, filtered and anonymized to make sure the queries we are collecting are both useful to the research community and respectful to our bing users and fans.
2. Using the sampled and anonymized queries Bing generates the 10 most relevant passages for the query.
3. Highly trained judges read the query and its related passages and if there is an answer present, the supporting passages are annotated and a natural language answer is generated.
4. A smaller proportion of queries(~17% of overall dataset with 182,887 unique queries) are then passed on to a second round of judges who are asked to verify the answer is correct and rewrite(if possible) the query to be a well formed answer. These answers are designed to be understood without perfect context and are designed with smart speakers/digital assistants in mind.

#### Data Format
Much like the v2.0 release, the v2.1 release is provided as a json file. This is for easy exploration and debugging and loading. Based on feedback from our community the V2.1 now dataset now has utilities for easy conversion to the [JSONL](http://jsonlines.org/) format. Official downloads from the website are as one large json object but use the tojson.py or tojsonl.py utilites to switch easy between file formats. 

Each line/entry containts the following parameters to be described below: query_id, query_type, query, passages, answers, and wellFormedAnswers.

1. query_id: A unique id for each query that is used in evaluation
2. query: A unique query based on initial Bing usage
3. passages: A set of 10:passages, URLs, and an annotation if they were used to formulate and answer(is_selected:1). Two passages may come from the URL and these passages have been obtained by Bing as the most relevant passages. If a passage is maked as is_selected:1 it means the judge used that passage to formulate their answer. If a passage is marked as is_selected:0 it means the judge did not use that passage to generate their answer. Questions that have the answer of 'No Answer Present.' will have all passages marked as is_selecte: 0.
4. query_type: A basic division of queries based on a trained classifier. Categories are:{LOCATION,NUMERIC,PERSON,DESCRIPTION,ENTITY} and can be used to debug model performance or make smaller more forcused datasets.
5. answers: An array of answers produced by human judges, most contain a single answer but ~1% contain more than one answer(average of ~2 answers if there are multiple answers). These answers were generated by real people in their own words instead of selecting a span of text. The language used in their answer may be similair or match the language in any of the passages.
6. wellFormedAnswers. An array of rewritten answers, most contain a single answer but ~1% contain more than one answer(average of ~5 answers if there are multiple answers). These answers were generated by having a new judge read the answer and the query and they would rewrite the answer if it did not (i) include proper grammar to make it a full sentence, (ii) make sense without the context of either the query or the passage, (iii) had a high overlap with exact portions in one of the context passages. This ensures that well formed answers are true natural languge and not just span selection. Well Formed Answers are a more difficult for of Question answering because they contain words that may not be present in either the question or any of the context passages. 

example
~~~
{
	"answers":["A corporation is a company or group of people authorized to act as a single entity and recognized as such in law."],
	"passages":[
		{
			"is_selected":0,
			"url":"http:\/\/www.wisegeek.com\/what-is-a-corporation.htm",
			"passage_text":"A company is incorporated in a specific nation, often within the bounds of a smaller subset of that nation, such as a state or province. The corporation is then governed by the laws of incorporation in that state. A corporation may issue stock, either private or public, or may be classified as a non-stock corporation. If stock is issued, the corporation will usually be governed by its shareholders, either directly or indirectly."},
		...
		}],
	"query":". what is a corporation?",
	"query_id":1102432,
	"query_type":"DESCRIPTION",
	"wellFormedAnswers":"[]"
}
~~~
## Utilities, Stats and Related Content
If you have any requests for utilities please open an issue and we will do our best to address it.
### Evaluation
Evaluation of systems will be done using the industry standard BLEU and ROUGE-L. These are far from perfect but have been the best option we have found that scales. If you know of a better metric or want to brainstorm creating one please contact us.

We have made the official evaluation script along with a sample output file on the dev set available for download as well so that you can evaluate your models. [Download the evaluation scripts](https://github.com/dfcf93/MSMARCOV2/tree/master/Evaluation) The evaluation script takes as inputs a reference and candidate output file. You can execute the evaluation script to evaluate your models as follows:
./run.sh <path to reference json file> <path to candidate json file> 
	
### Leaderboard Results
To Help Teams iterate we are making the results of official submissions on our evaluation script(the scores, not the full submissions)availible. We will update these files as we update metrics and as new submisions come in. They can be found in the [Leaderboard Results](https://github.com/dfcf93/MSMARCOV2/tree/master/Leaderboard%20Results) folder.


### Submissions
Once you have built a model that meets your expectations on evaluation with the dev set, you can submit your test results to get official evaluation on the test set. To ensure the integrity of the official test results, we do not release the correct answers for test set to the public. To submit your model for official evaluation on the test set, follow the below steps:
Run the evaluation script on the test set and generate the output results file for submission
Submit the following information by [contacting us](mailto:ms-marco@microsoft.com?subject=MS Marco Submission)
Individual/Team Name: Name of the individual or the team to appear in the leaderboard [Required]
Individual/Team Institution: Name of the institution of the individual or the team to appear in the leaderboard [Optional]
Model information: Name of the model/technique to appear in the leaderboard [Required]
Paper Information: Name, Citation, URL of the paper if model is from a published work to appear in the leaderboard [Optional]

Please submit your results either in json or jsonl format and ensure that each answer you are providing has its refrence query_id and query_text. If your model does not have query_id and query_text it is difficult/impossible to evalutate the submission.
To avoid "P-hacking" we discourage too many submissions from the same group in a short period of time. Because submissions don't require the final trained model we also retain the right to request a model to validate the results being submitted

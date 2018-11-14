# Ranking Task
MS MARCO(Microsoft Machine Reading Comprehension) is a large scale dataset focused on machine reading comprehension, question answering, and passage ranking. A variant of this task will be the part of [TREC 2019](https://trec.nist.gov/) and [AFIRM 2019](http://sigir.org/afirm2019/).

### Generation
To generate the ranking task dataset we started with the regular MSMARCO dataset which means if people want to generate any data in a different format they are more than able to(and even provide us with suggestions!). We are hoping to open source our production code shortly so people can generate the sets for themselves(with any normalization they may find interesting). 

We collected all unique passages(without any normalization) to make a pool of ~8.8 million unique passages. Then, for each query from the existing MSMARCO splits(train,dev, and eval) we ran a standard [BM25](https://en.wikipedia.org/wiki/Okapi_BM25) to produce 1000 relevant passages. These were ordered by random so each query now has 1000 corresponding passages. Since the original 10 passages presented with the query were extracted using the Bing ranking stack it possible that even none of the original passages are present with this new top 1000.

During the initial dataset creation, the judges would mark any passage that could answer the query which we then translated into our is_selected labels(relevant/used passages have is_selected=1). If a passage had is_selected=1 then this is a relevant query passage pair. It is worth noting that with these labels a positive is a true positive but negatives may not be a true negative(in other words there may ne relevant passages with is_selected=0). It is also worth noting that not all 1000 passages were seen by a judge and even ifWhile this means that it is possible for a system to find more relevant passages

To evaluate how well a system is reranking these 1000 relevant passages we use the already existing is_selected flag present in the v2.1 dataset. Given these labels on relevancy, a systems goal should be to rank any of the most relevant passage as high as possible. During the initial dataset creation, the judges would mark any passage that could answer the query which we then translated into our is_selected labels(relevant/used passages have is_selected=1). It is worth noting that with these labels a positive is a true positive but negatives may not be a true negative(in other words there may ne relevant passages with is_selected=0). It is also worth noting that not all 1000 passages were seen by a judge meaning it is possible that there are relevant passages that for purposed of this dataset are not considered relevant. 

Finally, understanding that this ranking data may not be useful to train a Deep Learning style system we build the triples files(availible in small and large(~27 and 
~270gb respectively)). These triples contain a query followed by a passage that has been marked as directly relevant(positive example) and another passage that has not been marked as relevant(negative). We understand that there could be a situtation where a one of the negative examples could actually be more relevant than the positive example but given the task goal is to rank the passages where we have a relevance passage as high as possible so this shouldn't be an issue. 

### Data, information, and Formating
Given that all files have been generated from the v2.1 dataset meaning that in theory anyone can generate the files we provide to their own specifications and liking. We will hopefully opensource our productin data shortly.
#### Passage to PassageID
This file contains 
#### Query to QueryID
This has been split for Train, Dev and Eval. 
#### Relevant Passages
We are releasing the Train and Dev
#### Triples.Train
The `triples.train.<size>.tsv` are two files that we have created as an easy to consume training dataset. Each line of the TSV contains querytext, A relevant passage, and an non-relevant passage all separated by `\t`. The only difference between triples.train.full.tsv and triples.train.small.tsv is the smaller is ~10% of the overall size since the full sized train is > 270gbs.

Example line:
```
what fruit is native to australia       Passiflora herbertiana. A rare passion fruit native to Australia. Fruits are green-skinned, white fleshed, with an unknown edible rating. Some sources list the fruit as edible, sweet and tasty, while others list the fruits as being bitter and inedible.assiflora herbertiana. A rare passion fruit native to Australia. Fruits are green-skinned, white fleshed, with an unknown edible rating. Some sources list the fruit as edible, sweet and tasty, while others list the fruits as being bitter and inedible.   The kola nut is the fruit of the kola tree, a genus (Cola) of trees that are native to the tropical rainforests of Africa.
```
  
#### Top1000
We have produced the Top1000 for Dev and for Eval
### Evaluation
Evaluation of systems will be done using MRR@10. We have selected such a low MRR number because the sizes of files candidates need to create quickly balloon with each additional depth. Official evaluation scripts and samples are availible [Here](https://github.com/dfcf93/MSMARCOV2/tree/master/Ranking/Evaluation).

### Submissions
Once you have built a model that meets your expectations on evaluation with the dev set, you can submit your test results to get official evaluation on the test set. To ensure the integrity of the official test results, we do not release the correct answers for test set to the public. To submit your model for official evaluation on the test set, follow the below steps:
Generate your proposed reranking for the Top1000 passages for the Eval set
Submit the following information by [contacting us](mailto:ms-marco@microsoft.com?subject=MS%20Marco%20Submission)

* Individual/Team Name: Name of the individual or the team to appear in the leaderboard [Required]
* Individual/Team Institution: Name of the institution of the individual or the team to appear in the leaderboard [Optional]
* Model information: Name of the model/technique to appear in the leaderboard [Required]
* Paper Information: Name, Citation, URL of the paper if model is from a published work to appear in the leaderboard [Optional]

To avoid "P-hacking" we discourage too many submissions from the same group in a short period of time. Because submissions don't require the final trained model we also retain the right to request a model to validate the results being submitted

##### Passage Reranking task Task
Given a query q and a the 1000 most relevant passages P = p1, p2, p3,... p1000, as retrieved by BM25 a succeful system is expected to rerank the most relevant passage as high as possible. For this task not all 1000 relevant items have a human labeled relevant passage. Evaluation will be done using [MRR](https://en.wikipedia.org/wiki/Mean_reciprocal_rank)

# Ranking Task
MS MARCO(Microsoft Machine Reading Comprehension) is a large scale dataset focused on machine reading comprehension, question answering, and passage ranking. 

### Generation
To generate the ranking task dataset we started with the regular MSMARCO dataset which means if people want to generate any data in a different format they are more than able to(and even provide us with suggestions!). We collected all unique passages(without any normalization) to make a pool of ~8.8 million unique passages. Then, for each query from the existing MSMARCO we ran a standard [BM25](https://en.wikipedia.org/wiki/Okapi_BM25) to produce 1000 relevant passages. These were ordered by random so each query now has 1000 corresponding passages. This was done for the train, dev, and evaluation sections of the v2.1 dataset. Understanding that this ranking data may not be useful to train a Deep Learning style system we build the triples.train.tsv files. These triples(availible in small and large(27 and 270gb respectively)) contain a query followed by a possitive passage and a negative passage. 

Then, the existing dataset has an annotation of is_selected:1 if a judge used a passage to generate their answer. We consider these as a ranking signla where all passages that have a value of 1 are a true possitive for query relevance for that given passage. Any passage that has a value of 0 is not a true negative. 
### Data, information, and Formating
Given that all files have been generated from the v2.1 dataset meaning that in theory anyone can generate the files we provide to their own specifications and liking.
#### Passage to PassageID

#### Query to QueryID
This has been split for Train, Dev and Eval. 
#### Relevant Passages
We are releasing the Train and Dev
#### Triples.Train
The triples.train.<size>.tsv are two files that we have created as an easy to consume training dataset. Each line of the TSV contains querytext, A relevant passage, and an non-relevant passage all separated by '\t'. The only difference between triples.train.full.tsv and triples.train.small.tsv is the smaller is ~10% of the overall size since the full sized train is > 270gbs.
'''
what fruit is native to australia       Passiflora herbertiana. A rare passion fruit native to Australia. Fruits are green-skinned, white fleshed, with an unknown edible rating. Some sources list the fruit as edible, sweet and tasty, while others list the fruits as being bitter and inedible.assiflora herbertiana. A rare passion fruit native to Australia. Fruits are green-skinned, white fleshed, with an unknown edible rating. Some sources list the fruit as edible, sweet and tasty, while others list the fruits as being bitter and inedible.   The kola nut is the fruit of the kola tree, a genus (Cola) of trees that are native to the tropical rainforests of Africa.
'''
#### Top1000
We have produced the Top1000 for Dev and for Eval
### Evaluation
Evaluation of systems will be done using MRR@10. We have selected such a low MRR number because the sizes of files candidates need to create quickly balloon with each additional depth. Official evaluation scripts and samples are availible [Here](https://github.com/dfcf93/MSMARCOV2/tree/master/Ranking/Evaluation).

### Submissions
Once you have built a model that meets your expectations on evaluation with the dev set, you can submit your test results to get official evaluation on the test set. To ensure the integrity of the official test results, we do not release the correct answers for test set to the public. To submit your model for official evaluation on the test set, follow the below steps:
Generate your proposed reranking for the Top1000 passages for the Eval set
Submit the following information by [contacting us](mailto:ms-marco@microsoft.com?subject=MS Marco Submission)
Individual/Team Name: Name of the individual or the team to appear in the leaderboard [Required]
Individual/Team Institution: Name of the institution of the individual or the team to appear in the leaderboard [Optional]
Model information: Name of the model/technique to appear in the leaderboard [Required]
Paper Information: Name, Citation, URL of the paper if model is from a published work to appear in the leaderboard [Optional]

To avoid "P-hacking" we discourage too many submissions from the same group in a short period of time. Because submissions don't require the final trained model we also retain the right to request a model to validate the results being submitted

##### Passage Reranking task Task
Given a query q and a the 1000 most relevant passages P = p1, p2, p3,... p1000, as retrieved by BM25 a succeful system is expected to rerank the most relevant passage as high as possible. For this task not all 1000 relevant items have a human labeled relevant passage. Evaluation will be done using [MRR](https://en.wikipedia.org/wiki/Mean_reciprocal_rank)

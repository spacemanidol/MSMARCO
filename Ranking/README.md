# Ranking Task
MS MARCO(Microsoft Machine Reading Comprehension) is a large scale dataset focused on machine reading comprehension, question answering, and passage ranking. A variant of this task will be the part of [TREC 2019](https://trec.nist.gov/) and [AFIRM 2019](http://sigir.org/afirm2019/).
## Passage Reranking task Task
Given a query q and a the 1000 most relevant passages P = p1, p2, p3,... p1000, as retrieved by BM25 a succeful system is expected to rerank the most relevant passage as high as possible. For this task not all 1000 relevant items have a human labeled relevant passage. Evaluation will be done using [MRR](https://en.wikipedia.org/wiki/Mean_reciprocal_rank)
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
This file contains each unique Passage in the larger MSMARCO dataset. Format is PID to Passage
````
7	Manhattan Project. The Manhattan Project was a research and development undertaking during World War II that produced the first nuclear weapons. It was led by the United States with the support of the United Kingdom and Canada. From 1942 to 1946, the project was under the direction of Major General Leslie Groves of the U.S. Army Corps of Engineers. Nuclear physicist Robert Oppenheimer was the director of the Los Alamos Laboratory that designed the actual bombs. The Army component of the project was designated the
8	In June 1942, the United States Army Corps of Engineersbegan the Manhattan Project- The secret name for the 2 atomic bombs.
9	One of the main reasons Hanford was selected as a site for the Manhattan Project's B Reactor was its proximity to the Columbia River, the largest river flowing into the Pacific Ocean from the North American coast.
````
Size info
````
8841823 collection.tsv
````
#### Query to QueryID
This has been split for Train, Dev and Eval. These sets include all queries including those which do not have answers. If queries with no answer were removed the sets would be around 35% smaller.
````
121352	define extreme
634306	what does chattel mean on credit history
920825	what was the great leap forward brainly
510633	tattoo fixers how much does it cost
737889	what is decentralization process.
278900	how many cars enter the la jolla concours d' elegance?
674172	what is a bank transit number
303205	how much can i contribute to nondeductible ira
570009	what are the four major groups of elements
492875	sanitizer temperature
````
Size info
````
  101093 queries.dev.tsv
  101092 queries.eval.tsv
  808731 queries.train.tsv
 1010916 total
````
#### Top1000
These files are split between train, dev, and eval. For each query there are ~1000 passages which were retrived by BM25 from the 8.8m collection. The train set contains all examples(~550,000 queries) but to make evaluation faster we have segmented the dev and eval file to be 1/8 of the full size. In other words, dev and email are ~6800 queries out of the 55000 possible.
````
188714  1000052 foods and supplements to lower blood sugar      Watch portion sizes: â  Even healthy foods will cause high blood sugar if you eat too much. â  Make sure each of your meals has the same amount of CHOs. Avoid foods high in sugar: â  Some foods to avoid: sugar, honey, candies, syrup, cakes, cookies, regular soda and.
1082792 1000084 what does the golgi apparatus do to the proteins and lipids once they arrive ?  Start studying Bonding, Carbs, Proteins, Lipids. Learn vocabulary, terms, and more with flashcards, games, and other study tools.
995526  1000094 where is the federal penitentiary in ind        It takes THOUSANDS of Macy's associates to bring the MAGIC of MACY'S to LIFE! Our associate team is an invaluable part of who we are and what we do. F ind the seasonal job that's right for you at holiday.macysJOBS.com!
199776  1000115 health benefits of eating vegetarian    The good news is that you will discover what goes into action spurs narrowing of these foods not only a theoretical supposition there are diagnosed with great remedy is said that most people and more can be done. Duncan was a wonderful can eating chicken cause gout benefits of natural. options with your health.
660957  1000115 what foods are good if you have gout?   The good news is that you will discover what goes into action spurs narrowing of these foods not only a theoretical supposition there are diagnosed with great remedy is said that most people and more can be done. Duncan was a wonderful can eating chicken cause gout benefits of natural. options with your health.
820267  1000130 what is the endocrine system responsible for?   The pancreas secretes pancreatic enzyme which is responsible for the breakdown of protein but it also secretes insulin so that we can get energy from glucose making it a part of the endocrine system or glands. The liver de-toxifies all of the blood that carries absorbed nutrients from the digestive system.
837202  1000252 what is the nutritional value of oatmeal        Oats make an easy, balanced breakfast. One cup of cooked oatmeal contains about 150 calories, four grams of fiber (about half soluble and half insoluble), and six grams of protein. To boost protein further, my favorite way to eat oatmeal is with a swirl of almond butter nestled within.
130825  1000268 definition for daring   Such a requirement would have three desirable consequences: First, it would tend to make bank executives more conservative and less daring in gambling with other people's money; second, it would put this liability of financial decision makers ahead of any taxpayer bailout in case of insolvency; and third, it would create a potentially powerful diseconomy of scale within big conglomerate banks.
408149  1000288 is dhgate a scam        If you think you ve been targeted by a counterfeit check scam, report it to the following agencies: 1  The Federal Trade Commission or 1-877-FTC-HELP (1-877-382-4357). 2  The U.S. Postal Inspection Service or call your local post office. 3  The number is in the Blue Pages of your local telephone directory.ere s how to avoid a counterfeit check scam: 1  Throw away any offer that asks you to pay for a prize or a gift. 2  If it s free or a gift, a promotion or a sweepstakes, you shouldn t have to pay anything for it.
345453  1000327 how to become a teacher assistant       Top 10 amazing movie makeup transformations. Biological chemistry, or biochemistry, is the study of the chemical composition of living organisms at a cellular level.omeone seeking a career in biological chemistry will usually need at least a bachelorâs degree. With a bachelorâs degree, an individual can qualify for a job as a science teacher at the high school level, a research assistant, laboratory technician, or a scientist in a testing environment.
Size info
````
   6668967 top1000.dev.tsv
   6515736 top1000.eval.tsv
  13184703 total
````
#### Relevant Passages
We have processed the train and dev set and made a QID to PID mapping of when a question has had a passage marked as relevant. We have held out the eval set but its distribution matches that of dev. As mentioned above, since since top1000.dev and top1000.eval are samples there exists qrels.dev.tsv(full qrels on 55,000 queries) and qrels.dev.small.tsv(which are the qrels corresponding to all queries in top1000.dev).
````
1185869	0
1185868	16
1185854	1176003
1185755	2205805
1184773	3214435
467556	5149343
44588	6986092
410717	6830906
1174761	754254
605123	965816
````
Size info
````
    7437 qrels.dev.small.tsv
   59273 qrels.dev.tsv
    7304 qrels.eval.small.tsv
   59187 qrels.eval.tsv
  532761 qrels.train.tsv
  665962 total
````
#### Triples.Train
The `triples.train.<size>.tsv` are two files that we have created as an easy to consume training dataset. Each line of the TSV contains querytext, A relevant passage, and an non-relevant passage all separated by `\t`. The only difference between triples.train.full.tsv and triples.train.small.tsv is the smaller is ~10% of the overall size since the full sized train is > 270gbs.

Example line:
````
what fruit is native to australia       Passiflora herbertiana. A rare passion fruit native to Australia. Fruits are green-skinned, white fleshed, with an unknown edible rating. Some sources list the fruit as edible, sweet and tasty, while others list the fruits as being bitter and inedible.assiflora herbertiana. A rare passion fruit native to Australia. Fruits are green-skinned, white fleshed, with an unknown edible rating. Some sources list the fruit as edible, sweet and tasty, while others list the fruits as being bitter and inedible.   The kola nut is the fruit of the kola tree, a genus (Cola) of trees that are native to the tropical rainforests of Africa.
````

### Evaluation
Evaluation of systems will be done using MRR@10. We have selected such a low MRR number because the sizes of files candidates need to create quickly balloon with each additional depth. Official evaluation scripts is [Here](https://github.com/dfcf93/MSMARCOV2/blob/master/Ranking/Baselines/msmarco_eval.py).
#### Rules
Since the Passage Reranking dataset is based on the original MSMARCO dataset it is possible to use some of the exisiting ranking signals in the original dataset as a relevance signal. In other words people can leverage the connection between the query and the 10 Bing passages in the original dataset and could be used to promote those passages or mine them for query expansion terms (relevance feedback). To prevent confusion of model performance we as any team that uses any signals from the initial dataset to describe what they used and we will mark the run as special. In addition, if you use any outside signal(or internal signal) that you think we should know and make know to the larger community please include a description in your submision. 

### Submissions
Once you have built a model that meets your expectations on evaluation with the dev set, you can submit your test results to get official evaluation on the test set. To ensure the integrity of the official test results, we do not release the correct answers for test set to the public. To submit your model for official evaluation on the test set, follow the below steps:
Generate your proposed reranking for the Top1000 passages for the Eval and the Dev set. To encourage reproducibility of results we encourage all teams to submit their code along with documentation and hyperparameters used.
Submit the following information by [contacting us](mailto:ms-marco@microsoft.com?subject=MS%20Marco%20Submission)
* Individual/Team Name: Name of the individual or the team to appear in the leaderboard [Required]
* Individual/Team Institution: Name of the institution of the individual or the team to appear in the leaderboard [Optional]
* Model information: Name of the model/technique to appear in the leaderboard [Required]
* Paper Information: Name, Citation, URL of the paper if model is from a published work to appear in the leaderboard [Optional]
* Code Information: A github repo of your model, instruction of how to use, etc [Optional]

To avoid "P-hacking" we discourage too many submissions from the same group in a short period of time. Because submissions don't require the final trained model we also retain the right to request a model to validate the results being submitted

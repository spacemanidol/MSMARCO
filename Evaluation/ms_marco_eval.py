"""
This module computes evaluation metrics for MS MaRCo data set.

For first time execution, please use run.sh to download necessary dependencies.
Command line:
/ms_marco_metrics$ PYTHONPATH=./bleu python ms_marco_eval.py <path_to_reference_file> <path_to_candidate_file>

Creation Date : 12/15/2018
Last Modified : 05/08/2018
Authors : Tri Nguyen <trnguye@microsoft.com>, Xia Song <xiaso@microsoft.com>, Tong Wang <tongw@microsoft.com>, Daniel Campos <dacamp@microsoft.com>
"""

from __future__ import print_function

import json
import sys
import spacy

from nltk.translate.bleu_score import sentence_bleu
from nltk.translate.bleu_score import SmoothingFunction
from rouge import Rouge 
from spacy.lang.en import English as NlpEnglish
nlp = spacy.load('en_core_web_lg') 
QUERY_ID_JSON_ID = 'query_id'
ANSWERS_JSON_ID = 'answers'
NLP = None
MAX_BLEU_ORDER = 4
YES_NO_DISCOUNT_RATE = 0.80

def normalize_batch(p_iter, p_batch_size=1000, p_thread_count=5):
    """Normalize and tokenize strings.

    Args:
    p_iter (iter): iter over strings to normalize and tokenize.
    p_batch_size (int): number of batches.
    p_thread_count (int): number of threads running.

    Returns:
    iter: iter over normalized and tokenized string.
    """

    global NLP
    if not NLP:
        NLP = NlpEnglish(parser=False)

    output_iter = NLP.pipe(p_iter, \
                           batch_size=p_batch_size, \
                           n_threads=p_thread_count)

    for doc in output_iter:
        tokens = [str(w).strip().lower() for w in doc]
        yield ' '.join(tokens)

def load_file(p_path_to_data):
    """Load data from json file.

    Args:
    p_path_to_data (str): path to file to load.
        File should be in format:
            {QUERY_ID_JSON_ID: <a_query_id_int>,
             ANSWERS_JSON_ID: [<list_of_answers_string>]}

    Returns:
    query_id_to_answers_map (dict):
        dictionary mapping from query_id (int) to answers (list of strings).
    no_answer_query_ids (set): set of query ids of no-answer queries.
    """

    all_answers = []
    query_ids = []
    no_answer_query_ids = set()
    with open(p_path_to_data, 'r', encoding='utf-8') as data_file:
        for line in data_file:
            try:
                json_object = json.loads(line)
            except json.JSONDecodeError:
                raise Exception('\"%s\" is not a valid json' % line)

            assert \
                QUERY_ID_JSON_ID in json_object, \
                '\"%s\" json does not have \"%s\" field' % \
                    (line, QUERY_ID_JSON_ID)
            query_id = json_object[QUERY_ID_JSON_ID]

            assert \
                ANSWERS_JSON_ID in json_object, \
                '\"%s\" json does not have \"%s\" field' % \
                    (line, ANSWERS_JSON_ID)
            answers = json_object[ANSWERS_JSON_ID]
            if 'No Answer Present.' in answers:
                no_answer_query_ids.add(query_id)
                answers = ['']
            all_answers.extend(answers)
            query_ids.extend([query_id]*len(answers))

    all_normalized_answers = normalize_batch(all_answers)

    query_id_to_answers_map = {}
    for i, normalized_answer in enumerate(all_normalized_answers):
        query_id = query_ids[i]
        if query_id not in query_id_to_answers_map:
            query_id_to_answers_map[query_id] = []
        query_id_to_answers_map[query_id].append(normalized_answer)
    return query_id_to_answers_map, no_answer_query_ids

def compute_metrics_from_files(p_path_to_reference_file,
                               p_path_to_candidate_file,
                               p_max_bleu_order):
    """Compute BLEU-N and ROUGE-L metrics.
    IMPORTANT: No-answer reference will be excluded from calculation.

    Args:
    p_path_to_reference_file (str): path to reference file.
    p_path_to_candidate_file (str): path to candidate file.
        Both files should be in format:
            {QUERY_ID_JSON_ID: <a_query_id_int>,
             ANSWERS_JSON_ID: [<list_of_answers_string>]}
    p_max_bleu_order: the maximum n order in bleu_n calculation.

    Returns:
    dict: dictionary of {'bleu_n': <bleu_n score>, 'rouge_l': <rouge_l score>}
    """

    reference_dictionary, reference_no_answer_query_ids = \
        load_file(p_path_to_reference_file)
    candidate_dictionary, candidate_no_answer_query_ids = load_file(p_path_to_candidate_file)
    query_id_answerable = set(reference_dictionary.keys())-reference_no_answer_query_ids
    query_id_answerable_candidate = set(candidate_dictionary.keys())-candidate_no_answer_query_ids
    
    true_positives = len(query_id_answerable_candidate.intersection(query_id_answerable))
    false_negatives = len(query_id_answerable)-true_positives
    true_negatives = len(candidate_no_answer_query_ids.intersection(reference_no_answer_query_ids))
    false_positives = len(reference_no_answer_query_ids)-true_negatives
    precision = float(true_positives)/(true_positives+false_positives) if (true_positives+false_positives)>0 else 1.
    recall = float(true_positives)/(true_positives+false_negatives) if (true_positives+false_negatives)>0 else 1.
    F1 = 2 *((precision*recall)/(precision+recall))
    filtered_reference_dictionary = \
        {key: value for key, value in reference_dictionary.items() \
                    if key not in reference_no_answer_query_ids}

    filtered_candidate_dictionary = \
        {key: value for key, value in candidate_dictionary.items() \
                    if key not in reference_no_answer_query_ids}

    for query_id, answers in filtered_candidate_dictionary.items():
        assert \
            len(answers) <= 1, \
            'query_id %d contains more than 1 answer \"%s\" in candidate file' % \
            (query_id, str(answers))

    reference_query_ids = set(filtered_reference_dictionary.keys())
    candidate_query_ids = set(filtered_candidate_dictionary.keys())
    common_query_ids = reference_query_ids.intersection(candidate_query_ids)
    assert (len(common_query_ids) == len(reference_query_ids)) and \
            (len(common_query_ids) == len(candidate_query_ids)), \
           'Reference and candidate files must share same query ids'

    similarity = 0
    bleu = [0,0,0,0]
    rouge_score = 0
    rouge = Rouge()
    smoothie = SmoothingFunction().method4

    for key in filtered_reference_dictionary:
        candidate_answer = filtered_candidate_dictionary[key][0]
        nlp_candidate_answer = nlp(candidate_answer)
        reference_answers = filtered_reference_dictionary[key]
        candidate_values = [0,0,0,0,0,0]
        selected_values = [0,0,0,0,0,0]
        for reference_answer in reference_answers:
            reference_split = reference_answer.split(',')
            candidate_values[0] = nlp_candidate_answer.similarity(nlp(reference_answer))
            candidate_values[1] = rouge.get_scores(candidate_answer, reference_answer)[0]['rouge-l']['f']
            candidate_values[2] = sentence_bleu(reference_answer, candidate_answer, weights=(1, 0, 0, 0), smoothing_function=smoothie)
            candidate_values[3] = sentence_bleu(reference_answer, candidate_answer, weights=(0.5,0.5,0,0), smoothing_function=smoothie)
            candidate_values[4] = sentence_bleu(reference_answer, candidate_answer, weights=(0.33,0.33,0.33,0), smoothing_function=smoothie)
            candidate_values[5] = sentence_bleu(reference_answer, candidate_answer, weights=(0.25,0.25,0.25,0.25), smoothing_function=smoothie)

            #partial credit for yes/no when complete answer is a yes/no question
            if (candidate_answer == 'yes' or candidate_answer == 'no') and (len(reference_split) > 1) and (candidate_answer == reference_split[0].strip()):
                for i in range(0,6):
                    selected_values[i] += max(candidate_values[i], YES_NO_DISCOUNT_RATE)
            else:
                for i in range(0,6):
                    selected_values[i] += candidate_values[i]

        similarity += (selected_values[0]/len(reference_answers))
        rouge_score += (selected_values[1]/len(reference_answers))
        for i in range (0,4):
            bleu[i] += (selected_values[i+2]/len(reference_answers))
        semantic_similarity = similarity/len(filtered_reference_dictionary)
    
    all_scores = {}
    all_scores['F1'] = F1
    all_scores['Semantic_Similarity'] = (semantic_similarity/len(filtered_reference_dictionary))
    all_scores['rouge_l'] = (rouge_score/len(filtered_reference_dictionary))
    for i in range(0,4):
        all_scores['bleu_%d' % (i+1)] = (bleu[i]/len(filtered_reference_dictionary))
    return all_scores

def main():
    """Command line: /ms_marco_metrics$ PYTHONPATH=./bleu python ms_marco_eval.py <path_to_reference_file> <path_to_candidate_file>"""

    path_to_reference_file = sys.argv[1]
    path_to_candidate_file = sys.argv[2]
    metrics = compute_metrics_from_files(path_to_reference_file, \
                                         path_to_candidate_file, \
                                         MAX_BLEU_ORDER)
    print('############################')
    for metric in sorted(metrics):
        print('%s: %s' % (metric, metrics[metric]))
    print('############################')

if __name__ == "__main__":
    main()

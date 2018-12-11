import sys

MAX_QUERY_TERMS = 80
MAX_DOC_TERMS = 400
regex_drop_char = re.compile('[^a-z0-9\s]+')
regex_multi_space = re.compile('\s+')

if __name__ == "__main__":
    if len(sys.argv) != 5:
        print("Usage: makeidf.py <train_triples_filename> <dev_rerank_filename> <eval_rerank_filename> <passage_collections_filename>")
        exit(-1)
python3 getMissingEmbeddings.py ~/Data/MSMARCO/ConversationalSearch/realQueries.tsv ~/Data/MSMARCO/ConversationalSearch/allQueries.tsv ~/Data/MSMARCO/ConversationalSearch/queryEmbeddings.tsv
wc -l  missing
cat missing>> ~/Data/MSMARCO/ConversationalSearch/queryEmbeddings.tsv
rm missing
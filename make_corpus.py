#%% Data Preparation
import re
import pandas as pd
from ContextMoralityIdeology.prepare_dataset import DATA_DIR
import spacy
from tqdm import tqdm
from pprint import pprint

link_regex = re.compile(r'<a href=\"/bill/[a-zA-Z\-/0-9]+\">[a-zA-Z\. 0-9]+</a>')
nlp = spacy.load('en_core_web_sm', disable=["tagger", 'ner'])


def parse_bill_links(row):
    text = row.speech
    # Uncomment to save links to mentioned bills
    #links = link_regex.findall(text)
    #row['bill_refs'] = links
    row.speech = link_regex.sub('<BILL>', text)
    return row


bill_item_indicator_re = re.compile(r'\([a-zA-Z0-9.]+\)')


def filter_sentence(sent_text):
    valid_sentence = all(['{time}' not in sent_text,
                          len(bill_item_indicator_re.findall(sent_text)) == 0,
                          len(sent_text.split()) > 2,
                          sent_text.upper() != sent_text,
                          not sent_text.startswith("By Mr"),
                          not sent_text.startswith("By Ms"),
                          not sent_text.startswith("A bill to"),
                          not (('yield' in sent_text or 'balance' in sent_text) and 'my time' in sent_text),
                          not (sent_text.startswith('(') and sent_text.endswith(')')),
                          not ('<BILL>' in sent_text and len(sent_text.split()) < 5),
                          not ('gentleman' in sent_text or 'gentlewoman' in sent_text) and not (
                                  'thanks' in sent_text or 'thank' in sent_text),
                          not (sent_text.startswith('Mr. Speaker') and 'yield' in sent_text)])

    return valid_sentence


speech_remove = re.compile(
    r'(?:Mr\.|Madam|The|The Acting) (?:Speaker|President|Vice President|Chair|Chairman),[\s]{1,}')


def clean_speech(speech_text):
    subbed = speech_remove.sub('', speech_text)
    return subbed



corpus = pd.read_csv(DATA_DIR + '/full_joined.tsv', '\t')
corpus = corpus[corpus.apply(lambda x: type(x.speech) != float, axis=1)]
corpus = corpus.apply(parse_bill_links, axis=1)
print(corpus)

#%% Clean the data
new_corpus = list()
doc_list = corpus.speech.tolist()
#%% Iterate through docs
for doc_raw in tqdm(doc_list, desc="Cleaning Speeches"):
    if len(doc_raw) > 100000:  # caught some bill transcripts
        new_corpus.append(list())
        continue
    cleaned_doc = clean_speech(doc_raw)
    doc_nlp = nlp(cleaned_doc)
    filtered_sentences = [sentence.text for sentence in doc_nlp.sents if filter_sentence(sentence.text)]
    if len(filtered_sentences) > 1:
        new_corpus.append(filtered_sentences)
    else:
        new_corpus.append(list())

# %% Process Cleaned
corpus["sentences"] = pd.Series(new_corpus, index=corpus.index)
corpus = corpus.dropna(subset=["sentences"]) \
    .drop(columns=['speech']) \
    .explode('sentences') \
    .dropna(subset=['sentences']) \
    .rename(columns={'sentences': 'sentence'})
corpus["bill_ref_count"] = corpus.sentence.apply(lambda x: x.count('BILL'))
print(corpus)
corpus.to_csv("./parsed_text_corpus.tsv", '\t', index=False)

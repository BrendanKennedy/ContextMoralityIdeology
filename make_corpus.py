import pandas as pd
import argparse, re

from moral import moral_words

parser = argparse.ArgumentParser()
parser.add_argument("--speeches", nargs='+', help="JSONL files of speeches")
args = parser.parse_args()

remove = re.compile(r"[^a-zA-Z\s]+")
alpha = re.compile(r"[a-zA-Z]{2,15}")

def clean(text):
    try:
        text = remove.sub("", text).lower()
        return " ".join(alpha.findall(text))
    except Exception as e:
        return ""

def remove_moral(text):
    return " ".join([w for w in text.split() if w not in moral_words])

def clean_field(field):
    return " ".join(field.split())

def expand(speech_df):
    dfs = list()
    for idx, row in speech_df.iterrows():
        date = row["date"]
        title = row["title"]
        url = row["url"]
        for speaker, speech in zip(row["Speakers"], row["Speeches"]):
            dfs.append({"date": date,
                "title": title, "url": url, "speaker": speaker,
                "speech": speech})
    return pd.DataFrame(dfs)

def get_speech_texts(paths):
    doc_dfs = list()
    for f in paths:
        section = "House" if f.startswith("house") else "Senate"
        for chunk in pd.read_json(f, orient='records', 
                lines=True, chunksize=10000):
            expanded = expand(chunk)
            expanded["speech"] = expanded["speech"].apply(clean)
            expanded["speaker"] = expanded["speaker"].apply(clean_field)
            expanded["url"] = expanded["url"].apply(clean_field)
            expanded["section"] = section
            doc_dfs.append(expanded)
            print(len(doc_dfs))
        corpus = pd.concat(doc_dfs, ignore_index=True)
    return corpus

if __name__ == '__main__':
    corpus = get_speech_texts(args.speeches)
    
    meta = corpus.loc[:, ["url", "speaker", "section", "date", "title"]]
    meta.index.name = "doc_num"
    meta.to_csv("metadata.tsv", sep='\t')

    #ddr = corpus.loc[:, ["speaker", "speech"]]
    #ddr.index.name = "doc_num"
    #ddr.to_csv("ddr_speeches.tsv", sep='\t')

    #lda = ddr.copy()
    #lda["speech"] = lda["speech"].apply(remove_moral)
    #lda.index.name = "doc_num"
    #lda.to_csv("lda_speeches.txt", sep='\t', header=None)

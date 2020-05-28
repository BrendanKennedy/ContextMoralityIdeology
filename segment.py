import numpy as np
import pandas as pd
import re
from tqdm import tqdm

def make_segment_regex(names):
    return re.compile(r'(?:The SPEAKER(?: pro tempore)?(?: \(M(?:r|s|rs)\. [a-zA-Z]+\) )?|The VICE PRESIDENT|M(?:r|s|rs)\. (?:{})(?: of [a-zA-Z]+\s*[a-zA-Z]*)?)[\.|\:]'.format("|".join(names)))


def segment(text, regex):
    if text is None or type(text) == float:
        return list(), list()
    speakers = regex.findall(text)
    segments = regex.split(text)
    if len(segments) != len(speakers) + 1:
        segments = list(); speakers = list()
    else:
        segments = segments[1:]
    return speakers, segments

def parse_raw_text(senate_or_house="senate"):
    speakers = pd.read_json("people/{}_members.json".format(senate_or_house),
                            orient='records',
                            lines=True)
    last_names = set(speakers["LastName"].values)
    regex = make_segment_regex(last_names)

    meta = pd.read_json("{}_meta.json".format(senate_or_house),
                        orient='records',
                        lines=True)\
        .set_index('url')
    docs = pd.read_json("{}_docs.json".format(senate_or_house),
                        orient='records',
                        lines=True)\
        .set_index('url')\
        .join(meta, how='inner')\
        .dropna(subset=["body"])\
        .iloc[:1000,:]

    results = list()  # df
    for url, row in tqdm(docs.iterrows(),
                         total=len(docs)):
        speakers, text_segments = segment(row["body"], regex)
        for speaker, text_segment in zip(speakers, text_segments):
            results.append({'speaker': speaker,
                            'segment': text_segment,
                            'url': url,
                            'session': row['session'],
                            'date': row['date']})
    speeches = pd.DataFrame(results)
    return speeches

    """
    people = pd.read_csv("matched.tsv", delimiter='\t')
    people.columns = ["doc_num"] + list(people.columns)[1:]
    people = people.loc[:, ["doc_num", "State", "Party", "Name", "District"]]
    people.index = people["doc_num"]

    matched_doc_nums = people["doc_num"].values.tolist()

    merged = speeches.join(people, how='inner')
    merged.to_json("{}_merged.json".format(group), orient='records', lines=True)
    """

def extract_speaker(segment_df_row):
    name = segment_df_row["speaker"]
    address = name.split()[0]
    gender = "M" if address == "Mr." else "F"
    name = name.strip()
    name = name.strip('.').strip(':')
    tokens = name.split()
    if "of" in tokens:
        idx = tokens.index("of")
        state_str = " ".join(tokens[idx+1:])
        tokens = tokens[:idx]
    else:
        state_str = None
    name = " ".join(tokens[1:])
    return pd.Series({"speaker_name": name,
                      "speaker_state": state_str,
                      "year": segment_df_row["date"].strftime('%Y'),
                      "speaker_sex": gender,
                      "speaker_session": segment_df_row["session"]})


def merge_speaker_info(row):
    name = row["Name"]
    section = row["Section"]
    if section == "House":
        name_match = house_speakers[house_speakers["LastName"] == name]
    elif section == "Senate":
        name_match = senate_speakers[senate_speakers["LastName"] == name]
    if len(name_match) == 1:
        return pd.Series(name_match.iloc[0,:].to_dict())

    year = int(row["Year"])
    year_mask = name_match[section].apply(lambda x: year in x)
    year_match = name_match[year_mask]
    if len(year_match) == 1:
        return pd.Series(year_match.iloc[0,:].to_dict())
    if row["State"] is not None:
        state_match = year_match[year_match["State"] == row["State"]]
        if len(state_match) == 1:
            return pd.Series(state_match.iloc[0,:].to_dict())
        return pd.Series(state_match.to_dict())
    return pd.Series(year_match.to_dict())



if __name__ == '__main__':

    house_df = parse_raw_text('house')
    house_df = house_df.join(house_df.apply(extract_speaker, axis=1))
    print(house_df['year'])
    #senate_df = parse_raw_text('senate')

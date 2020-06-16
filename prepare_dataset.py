import numpy as np
import os
import pandas as pd
import re
from tqdm import tqdm

pd.set_option('display.max_columns', 500)
desired_width = 500
pd.set_option('display.width', desired_width)

DATA_DIR = "/Volumes/GoogleDrive/My Drive/ContextMoralityIdeology/data/"


def read_people_db_txt():
    path = os.path.join(DATA_DIR, 'people', 'people_rawtext.txt')
    people = list()
    with open(path) as fo:
        buffer = dict()
        for line in fo:
            if len({"Representative", "Senator"}.intersection(set(line.split()))) > 0:
                if len(buffer) == 0:
                    buffer["Name"] = " ".join(line.split()[2:])
                else:
                    people.append(buffer)
                    buffer = dict()
                    buffer["Name"] = " ".join(line.split()[2:])

            elif line.startswith("State"):
                buffer["State"] = line.strip()
            elif line.startswith("District"):
                buffer["District"] = line.strip()
            elif line.startswith("Party"):
                buffer["Party"] = line.strip()
            elif line.startswith("Served"):
                continue
            elif line.startswith("House"):
                buffer["HouseTerm"] = line.strip()
            elif line.startswith("Senate"):
                buffer["SenateTerm"] = line.strip()
            else:
                continue

    df = pd.DataFrame(people)
    df.State = df.State.apply(lambda s: s.replace("State: ", ""))
    df.Party = df.Party.apply(lambda s: s.replace("Party: ", ""))

    def expand_years(year_string):
        if type(year_string) == float:
            return np.NaN
        years = list()
        term = year_string.replace("Senate: ", "").replace("House: ", "")
        term = term.replace("Present", "2019")
        terms = term.split(',')
        for t in terms:
            if '-' not in t:
                years.append(t)
                continue
            start, end = [int(v) for v in t.split('-')]
            for i in range(start, end + 1):
                years.append(i)
        return years

    df.SenateTerm = df.SenateTerm.apply(expand_years)
    senate_years = df.SenateTerm.apply(lambda x: pd.Series(x)).unstack()
    s = df.drop('SenateTerm', axis=1) \
        .join(pd.DataFrame(senate_years.reset_index(level=0, drop=True).dropna()), how='inner') \
        .rename(columns={0: "year"}) \
        .assign(level='senate') \
        .drop('HouseTerm', axis=1)
    df.HouseTerm = df.HouseTerm.apply(expand_years)
    house_years = df.HouseTerm.apply(lambda x: pd.Series(x)).unstack()
    h = df.drop('HouseTerm', axis=1) \
        .join(pd.DataFrame(house_years.reset_index(level=0, drop=True).dropna()), how='inner') \
        .rename(columns={0: "year"}) \
        .assign(level='house') \
        .drop('SenateTerm', axis=1)

    df = pd.concat((s, h), axis=0, ignore_index=True)

    def parse_name(name):
        last = name.split(',')[0]
        uppercopy = list(last.upper())
        prev = last[1]
        flag = False
        for i, char in enumerate(last[2:]):
            if char.isupper() and prev.isalpha():
                flag = True
                for j in range(1, i + 2):
                    uppercopy[j] = uppercopy[j].lower()
                continue
            prev = last[i + 2]
        return "".join(uppercopy)

    df["LastName"] = df.Name.apply(parse_name)
    df.year = df.year.apply(lambda x: str(int(x)))
    return df


def make_segment_regex(names):
    return re.compile(
        r'(?:A recorded vote was ordered|The (?:CHAIRMAN|SPEAKER|Acting CHAIR)(?: pro tempore)?(?: \(M(?:r|s|rs)\. [a-zA-Z]+\) )?|The VICE PRESIDENT|M(?:r|s|rs)\. (?:{})(?: of [a-zA-Z]+\s*[a-zA-Z]*)?)[\.|\:]'.format(
            "|".join(names)))


def segment(text, regex):
    if text is None or type(text) == float:
        return list(), list()
    speakers = regex.findall(text)
    segments = regex.split(text)
    if len(segments) != len(speakers) + 1:
        segments = list()
        speakers = list()
    else:
        segments = segments[1:]
    return speakers, segments


def parse_raw_text(senate_or_house="senate"):
    speakers = pd.read_json(DATA_DIR + "people/{}_members.json".format(senate_or_house),
                            orient='records',
                            lines=True)
    last_names = set(speakers["LastName"].values)
    regex = make_segment_regex(last_names)

    meta = pd.read_json(DATA_DIR + "{}_meta.json".format(senate_or_house),
                        orient='records',
                        lines=True) \
        .set_index('url')
    docs = pd.read_json(DATA_DIR + "{}_docs.json".format(senate_or_house),
                        orient='records',
                        lines=True)\
        .set_index('url') \
        .join(meta, how='inner') \
        .dropna(subset=["body"])

    results = list()  # df
    for url, row in tqdm(docs.iterrows(),
                         desc="Processing {}".format(senate_or_house),
                         total=len(docs)):
        speakers, text_segments = segment(row["body"], regex)
        for speaker, text_segment in zip(speakers, text_segments):
            results.append({'speaker': speaker,
                            'speech': re.sub(r'[\s]+', ' ', text_segment).strip(),
                            'url': url,
                            'level': senate_or_house,
                            'session': row['session'],
                            'date': row['date']})
    speeches = pd.DataFrame(results)
    return speeches


def extract_speaker(segment_df_row):
    name = segment_df_row["speaker"]
    address = name.split()[0]
    gender = "M" if address == "Mr." else "F"
    name = name.strip()
    name = name.strip('.').strip(':')
    tokens = name.split()
    if "of" in tokens:
        idx = tokens.index("of")
        state_str = " ".join(tokens[idx + 1:])
        tokens = tokens[:idx]
    else:
        state_str = None
    name = " ".join(tokens[1:])
    return pd.Series({"speaker_name": name,
                      "speaker_state": state_str,
                      "year": segment_df_row["date"].strftime('%Y'),
                      "speaker_sex": gender,
                      "speaker_session": segment_df_row["session"]})


def merge_speaker_info(text_df, people_df):
    text_df = text_df[text_df.speaker_name != 'VICE PRESIDENT']
    text_df = text_df[text_df.speaker_name != 'PRESIDENT']
    text_df = text_df[text_df.speaker_name != 'SPEAKER pro tempore']
    people_df.level = people_df.level.apply(lambda x: x.lower())
    merged_df = text_df.merge(people_df,
                              how='inner',
                              left_on=('speaker_name', 'year', 'level'),
                              right_on=('LastName', 'year', 'level'))
    return merged_df


if __name__ == '__main__':
    people_df = read_people_db_txt()

    senators = people_df[people_df.level == 'senate']
    house = people_df[people_df.level == 'house']

    senate_df = parse_raw_text('senate')
    senate_df = senate_df.join(senate_df.apply(extract_speaker, axis=1))
    senate_df = merge_speaker_info(senate_df, people_df)

    house_df = parse_raw_text('house')
    house_df = house_df.join(house_df.apply(extract_speaker, axis=1))
    house_df = merge_speaker_info(house_df, people_df)

    df = pd.concat((house_df, senate_df), axis=0, ignore_index=True)\
        .dropna(subset=['Name'])\
        .drop(columns=['speaker', 'url', 'session', 'speaker_session', 'speaker_state',
                       'speaker_name', 'year', 'LastName', 'District'])\
        .rename(columns={'level': 'LegislativeBody', 'speaker_sex': 'Gender'})
    df.LegislativeBody = df.LegislativeBody.map({'house': 'House', 'senate': 'Senate'})
    df.Gender = df.Gender.map({'M': 'Male', 'F': 'Female'})
    df.to_csv(DATA_DIR + "/full_joined.tsv", '\t', index=False)

# coding: utf-8

import pandas as pd
from datetime import datetime
from copy import deepcopy

#meta = pd.read_csv("metadata.tsv", delimiter='\t')
#house_speakers = pd.read_json("people/house_members.json", orient='records', lines=True)
#senate_speakers = pd.read_json("people/senate_members.json", orient='records', lines=True)


parsed = meta.apply(parse, axis=1)
doc_nums = parsed["doc_num"].values.tolist()

merged = parsed.apply(merge, axis=1)
merged.index = doc_nums

merged["is_unique"] = merged["State"].apply(lambda x: not isinstance(x, dict))
matched = merged[merged["is_unique"]]
matched.drop(columns=["is_unique"], inplace=True)
matched.to_csv("matched.tsv", sep='\t')

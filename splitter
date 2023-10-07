#!/usr/bin/env python
# -*- coding: utf-8 -*-

import argparse
import functions as f
import pandas as pd

parser = argparse.ArgumentParser(description='Change - Produces a report of the significant feature changes')
parser.add_argument('subject')
parser.add_argument('column')
parser.add_argument('--df2', required=False)
known, unknown = parser.parse_known_args()
known = {k: v for k, v in vars(known).items() if v is not None}
unknown = eval(unknown[0]) if unknown != [] else {}

# need to sort this out for the output
df = pd.read_csv(f'../results/{known.get("subject")}.tsv', sep='\t', index_col=0)
if known.get("df2"):
    df2 = pd.read_csv(f'../results/{known.get("df2")}.tsv', sep='\t' ,index_col=0)
else
    df2 = pd.read_csv(f'../results/meta.tsv', sep='\t' ,index_col=0)
output = f.splitter(df, df2, **known|unknown)
print(output.to_string())
for col in output.columns:
    output[col].to_csv(f'../results/{subject}{col}{level}.tsv', sep='\t')

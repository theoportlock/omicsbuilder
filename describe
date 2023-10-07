#!/usr/bin/env python
# -*- coding: utf-8 -*-

import argparse
import functions as f
import pandas as pd

parser = argparse.ArgumentParser(description='Describe - Produces a summary report of analysis')
parser.add_argument('subject')
parser.add_argument('-p', '--pval', type=float)
parser.add_argument('-c', '--change')
parser.add_argument('-s', '--sig')
parser.add_argument('-r', '--corr', action='store_true')
known, unknown = parser.parse_known_args()
known = {k: v for k, v in vars(known).items() if v is not None}
unknown = eval(unknown[0]) if unknown != [] else {}

df = pd.read_csv(f'../results/{known.get("subject")}.tsv', sep='\t', index_col=0)
output = f.describe(df, **known|unknown)
print(output.to_string())
output.to_csv(f'../results/{known.get("subject")}describe.tsv', sep='\t', index=False)


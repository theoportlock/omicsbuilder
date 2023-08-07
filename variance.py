#!/usr/bin/env python
# -*- coding: utf-8 -*-

import argparse
import functions as f
import pandas as pd

parser = argparse.ArgumentParser(description='Calculates the explained variance of the categorical labels of the metadata')
parser.add_argument('-df1', required=True)
parser.add_argument('-df2', required=False)
parser.add_argument('-p', '--pval')
known, unknown = parser.parse_known_args()
known = {k: v for k, v in vars(known).items() if v is not None}
unknown = eval(unknown[0]) if unknown != [] else {}

if args.get('df2'):
    DF1 = pd.read_csv(f'../results/{known.get("df1")}.tsv', sep='\t', index_col=0).dropna()
    DF2 = pd.read_csv(f'../results/{known.get("df2")}.tsv', sep='\t', index_col=0).dropna()
    output = f.explainedvariance(DF1, DF2, **known|unknown)
else:
    DF1 = pd.read_csv(f'../results/{known.get("df1")}.tsv', sep='\t', index_col=0).dropna()
    output = f.variance(DF1, **known|unknown)
print(output.to_string())
output.to_csv(f'../results/{df1}{df2}power.tsv', sep='\t')

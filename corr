#!/usr/bin/env python
# -*- coding: utf-8 -*-

import argparse
import functions as f
import pandas as pd

parser = argparse.ArgumentParser(description='Corr - Produces a report of the significant correlations between data')
parser.add_argument('subject')
parser.add_argument('-m', '--mult', action='store_true')
known, unknown = parser.parse_known_args()
known = {k: v for k, v in vars(known).items() if v is not None}
unknown = eval(unknown[0]) if unknown != [] else {}

df = pd.read_csv(f'../results/{known.get("subject")}.tsv', sep='\t', index_col=0)
output = f.corr(df, **known|unknown)
print(output)
output.to_csv(f'../results/{known.get("subject")}corr.tsv', sep='\t')

#!/usr/bin/env python
# -*- coding: utf-8 -*-

import argparse
import functions as f
import pandas as pd

parser = argparse.ArgumentParser(description='Change - Produces a report of the significant feature changes')
parser.add_argument('subject')
parser.add_argument('-a', '--analysis', nargs='+', default=['prevail','mww','lfc'])
parser.add_argument('--mult', action='store_true')
known, unknown = parser.parse_known_args()
known = {k: v for k, v in vars(known).items() if v is not None}
unknown = eval(unknown[0]) if unknown != [] else {}

df = pd.read_csv(f'../results/{known.get("subject")}.tsv', sep='\t', index_col=0)
output = f.change(df, **known|unknown)
print(output.to_string())
output.to_csv(f'../results/{known.get("subject")}change.tsv', sep='\t')


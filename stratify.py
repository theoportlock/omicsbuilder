#!/usr/bin/env python
# -*- coding: utf-8 -*-

import argparse
import functions as f
import pandas as pd

parser = argparse.ArgumentParser(description='Change - Produces a report of the significant feature changes')
parser.add_argument('subject')
parser.add_argument('level')
parser.add_argument('--df2')
known, unknown = parser.parse_known_args()
known = {k: v for k, v in vars(args).items() if v is not None}
unknown = eval(unknown[0]) if unknown != [] else {}

df = pd.read_csv(f'../results/{known.get("subject")}.tsv', sep='\t', index_col=0)
meta = pd.read_csv(f'../results/{known.get("df2")}.tsv', sep='\t' ,index_col=0)
level = known.get('level')
output = f.stratify(df, meta, level **known|unknown)
print(output.to_string())
output.to_csv(f'../results/{known.get("subject")}{level}.tsv', sep='\t')

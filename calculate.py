#!/usr/bin/env python
# -*- coding: utf-8 -*-

import argparse
import functions as f

parser = argparse.ArgumentParser(description='Calculate - compute a value for each sample based on features')
parser.add_argument('analysis')
parser.add_argument('subject')
known, unknown = parser.parse_known_args()
known = {k: v for k, v in vars(args).items() if v is not None}
unknown = eval(unknown[0]) if unknown != [] else {}

df = pd.read_csv(f'../results/kwargs.get("subject").tsv', sep='\t', index_col=0)
output = f.calculate(df, **known|unknown)
print(output.to_string())
output.to_csv(f'../results/{kwargs.get("subject")}{kwargs.get("analysis")}.tsv', sep='\t')

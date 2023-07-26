#!/usr/bin/env python
# -*- coding: utf-8 -*-

import argparse
import functions as f

parser = argparse.ArgumentParser(description='Filter')
parser.add_argument('subject')
parser.add_argument('-lt', type=float, help='threshold value upper bound for filtering')
parser.add_argument('-gt', type=float, help='threshold value lower bound for filtering')
parser.add_argument('-c', '--column', type=str, help='column for thesholding')
parser.add_argument('-n', '--name', type=str, help='name for filtering')
parser.add_argument('-rf', '--rowfilt', type=str, help='regex for index filtering')
parser.add_argument('-cf', '--colfilt', type=str, help='regex for column filtering')
parser.add_argument('-m', '--min_unique', type=int)
parser.add_argument('-fdf', '--filter_df')
parser.add_argument('-fdfx', '--filter_df_axis', type=int)
parser.add_argument('-absgt', type=float)
known, unknown = parser.parse_known_args()
known = {k: v for k, v in vars(args).items() if v is not None}
unknown = eval(unknown[0]) if unknown != [] else {}

df = pd.read_csv(f'../results/{known.get("subject")}.tsv', sep='\t', index_col=0)
if known.get("filter_df"):
    known['filter_df'] = pd.read_csv(f'../results/{known.get("filter_df")}.tsv', sep='\t', index_col=0)
output = f.filter(df, **known|unknown)
print(output.to_string())
output.to_csv(f'../results/{known.get("subject")}filter.tsv', sep='\t')

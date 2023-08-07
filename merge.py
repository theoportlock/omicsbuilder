#!/usr/bin/env python
# -*- coding: utf-8 -*-

import argparse
import functions as f
import pandas as pd

parser = argparse.ArgumentParser(description='Merge - Combines datasets')
parser.add_argument('datasets', nargs='+')
parser.add_argument('-t', '--type')
parser.add_argument('-a', '--append', action='store_true')
parser.add_argument('-f', '--filename')
known, unknown = parser.parse_known_args()
known = {k: v for k, v in vars(known).items() if v is not None}
unknown = eval(unknown[0]) if unknown != [] else {}

dfs = known.get("datasets"):
output = f.merge(dfs, **known|unknown)
print(output.to_string())

if known.get("filename"):
    outdf.to_csv(f'../results/{filename}.tsv', sep='\t')
else:
    outdf.to_csv(f'../results/{"".join(dfs)}.tsv', sep='\t')

#!/usr/bin/env python
# -*- coding: utf-8 -*-

import argparse
import pandas as pd

def merge(datasets=None, type='inner'):
    outdf = pd.concat([pd.read_csv(f'../results/{subject}.tsv', sep='\t', index_col=0) for subject in datasets], axis=1, join=type)
    outdf.to_csv(f'../results/{"".join(datasets)}.tsv', sep='\t')
    return outdf


if __name__ == '__main__':
    parser = argparse.ArgumentParser(description='Merge - Combines datasets')
    parser.add_argument('datasets', nargs='+')
    parser.add_argument('-t', '--type')
    args = parser.parse_args()
    args = {k: v for k, v in vars(args).items() if v is not None}
    print(args)
    output = merge(**args)
    print(output)

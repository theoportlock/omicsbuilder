#!/usr/bin/env python3
# -*- coding: utf-8 -*-
import pandas as pd
import argparse

def stratify(subject, level=None):
    '''
    If no arguments given then just stratify by all metadata
    '''
    df = pd.read_csv(f'../results/{subject}.tsv', sep='\t', index_col=0)
    meta = pd.read_csv("../results/meta.tsv", sep='\t' ,index_col=0)
    if level:
        metadf = df.join(meta[level].dropna(), how='inner').reset_index(drop=True).set_index(level)
        metadf.to_csv(f'../results/{subject}{level}.tsv', sep='\t')
    else:
        metadf = []
        for level in meta.columns:
            merge = df.join(meta[level].dropna(), how='inner').reset_index(drop=True).set_index(level)
            metadf.append(merge)
            merge.to_csv(f'../results/{subject}{level}.tsv', sep='\t')
    return metadf

if __name__ == '__main__':
    parser = argparse.ArgumentParser(description='Change - Produces a report of the significant feature changes')
    parser.add_argument('subject')
    parser.add_argument('-l', '--level')
    args = parser.parse_args()
    stratify(args.subject, args.level)

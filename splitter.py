#!/usr/bin/env python
# -*- coding: utf-8 -*-
import pandas as pd
import argparse

def splitter(subject, column, df2='meta'):
    '''
    If no arguments given then just stratify by all metadata
    '''
    df = pd.read_csv(f'../results/{subject}.tsv', sep='\t', index_col=0)
    meta = pd.read_csv(f"../results/{df2}.tsv", sep='\t' ,index_col=0)
    metadf = []
    for level in meta[column].unique():
        merge = df.join(meta.loc[meta[column] == level,column], how='inner').drop(column, axis=1)
        metadf.append(merge)
        merge.to_csv(f'../results/{subject}{column}{level}.tsv', sep='\t')
    return metadf

if __name__ == '__main__':
    parser = argparse.ArgumentParser(description='Change - Produces a report of the significant feature changes')
    parser.add_argument('subject')
    parser.add_argument('column')
    parser.add_argument('--df2', required=False)
    args = parser.parse_args()
    args = {k: v for k, v in vars(args).items() if v is not None}
    output = splitter(**args)
    print(output)


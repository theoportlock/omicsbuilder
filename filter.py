#!/usr/bin/env python
# -*- coding: utf-8 -*-

import argparse
import numpy as np
import pandas as pd

def filter(subject, min_unique=None, gt=None, lt=None, column=None, filter_df=None, filter_df_axis=0, absgt=None, rowfilt=None, colfilt=None):
    df = pd.read_csv(f'../results/{subject}.tsv', sep='\t', index_col=0)
    if filter_df:
        fdf = pd.read_csv(f'../results/{filter_df}.tsv', sep='\t', index_col=0)
        if filter_df_axis == 1:
            df = df.loc[:, fdf.index]
        else:
            df = df.loc[fdf.index]
    if colfilt:
        df = df.loc[:, df.columns.str.contains(colfilt, regex=True)]
    if rowfilt:
        df = df.loc[df.index.str.contains(rowfilt, regex=True)]
    if min_unique:
        '''
        df = df.loc[
                df.agg(np.count_nonzero, axis=1) > min_unique,
                df.agg(np.count_nonzero, axis=0) > min_unique]
        '''
        df = df.loc[:, df.agg(np.count_nonzero, axis=0) > min_unique]
    if column and lt:
        df = df.loc[df[column].lt(lt)]
    elif lt:
        df = df.loc[:, df.abs().lt(lt).any(axis=0)]
    if column and gt:
        df = df.loc[df[column].gt(gt)]
    elif gt:
        df = df.loc[:, df.abs().gt(gt).any(axis=0)]
    if column and absgt:
        df = df.loc[df[column].abs().gt(absgt)]
    df.to_csv(f'../results/{subject}filter.tsv', sep='\t')
    return df


if __name__ == '__main__':
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
    args = parser.parse_args()
    args = {k: v for k, v in vars(args).items() if v is not None}
    output = filter(**args)
    print(output)

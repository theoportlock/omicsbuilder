#!/usr/bin/env python3
# -*- coding: utf-8 -*-

import pandas as pd
import argparse
import numpy as np

def filter(subject, min_unique=0, gt=None, lt=None, column=None):
    df = pd.read_csv(f'../results/{subject}.tsv', sep='\t', index_col=0)
    df = df.loc[
            df.agg(np.count_nonzero, axis=1) > min_unique,
            df.agg(np.count_nonzero, axis=0) > min_unique]
    if column and lt:
        df = df.loc[df[column].lt(lt)]
    elif lt:
        df = df.loc[:, df.abs().lt(lt).any(axis=0)]
    if column and gt:
        df = df.loc[df[column].gt(gt)]
    elif gt:
        df = df.loc[:, df.abs().gt(gt).any(axis=0)]
    df.to_csv(f'../results/{subject}filter.tsv', sep='\t')
    return df


if __name__ == '__main__':
    parser = argparse.ArgumentParser(description='Filter')
    parser.add_argument('subject')
    parser.add_argument('-lt', type=float, help='threshold value upper bound for filtering')
    parser.add_argument('-gt', type=float, help='threshold value lower bound for filtering')
    parser.add_argument('-c', '--column', type=str, help='column for thesholding')
    parser.add_argument('-m', '--min-unique', type=int)
    args = parser.parse_args()
    args = {k: v for k, v in vars(args).items() if v is not None}
    output = filter(**args)
    print(*output)

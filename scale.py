#!/usr/bin/env python
# -*- coding: utf-8 -*-

from skbio.stats.composition import clr
from skbio.stats.composition import multiplicative_replacement as mul
from sklearn.preprocessing import MinMaxScaler
from sklearn.preprocessing import StandardScaler
import argparse
import pandas as pd
import numpy as np

def norm(df):
    return df.T.div(df.sum(axis=1), axis=1).T

def standard(df):
    scaledDf = pd.DataFrame(
            StandardScaler().fit_transform(df.T),
            index=df.T.index,
            columns=df.T.columns).T
    return scaledDf

def minmax(df):
    scaledDf = pd.DataFrame(
            MinMaxScaler().fit_transform(df),
            index=df.index,
            columns=df.columns)
    return scaledDf

def log(df):
    return df.apply(np.log1p)

def clr(df):
    return pd.DataFrame(clr(df), index=df.index, columns=df.columns)

def mult(df):
    return pd.DataFrame(mul(df), index=df.index, columns=df.columns)

def scale(analysis, subject, **kwargs):
    df = pd.read_csv(f'../results/{subject}.tsv', sep='\t', index_col=0)
    available={
        'norm':norm,
        'standard':standard,
        'minmax':minmax,
        'log':log,
        'clr':clr,
        'mult':mult,
        }
    output = available.get(analysis)(df, **kwargs)
    output.to_csv(f'../results/{subject}{analysis}.tsv', sep='\t')
    return output

if __name__ == '__main__':
    parser = argparse.ArgumentParser(description='Calculate - compute a value for each sample based on features')
    parser.add_argument('analysis')
    parser.add_argument('subject')
    args, unknown = parser.parse_known_args()
    args = {k: v for k, v in vars(args).items() if v is not None}
    kwargs = eval(unknown[0]) if unknown != [] else {}
    output = scale(**args|kwargs)
    print(output)

#!/usr/bin/env python3
# -*- coding: utf-8 -*-

from skbio.stats.composition import clr
from sklearn.preprocessing import MinMaxScaler
from sklearn.preprocessing import StandardScaler
import pandas as pd
import pandas as pd
import argparse
import matplotlib.pyplot as plt
import pandas as pd
from skbio.stats.composition import multiplicative_replacement as mul
import sys
import utils

def norm(df):
    return df.T.div(df.sum(axis=1), axis=1).T

def Standard(df):
    scaledDf = pd.DataFrame(
            StandardScaler().fit_transform(df),
            index=df.index,
            columns=df.columns)
    return scaledDf

def minmax(df):
    scaledDf = pd.DataFrame(
            MinMaxScaler().fit_transform(df),
            index=df.index,
            columns=df.columns)
    return scaledDf

def log(df):
    return df.apply(np.log1p)

def clr(df, axis=0):
    return pd.DataFrame(clr(df), index=df.index, columns=df.columns)

def mult(df):
    return pd.DataFrame(mul(df), index=df.index, columns=df.columns)


if __name__ == '__main__':
    parser = argparse.ArgumentParser(description='Filter')
    parser.add_argument('subject')
    parser.add_argument('-p', '--pvalthresh', type=float)
    args = parser.parse_args()
    args = {k: v for k, v in vars(args).items() if v is not None}
    output = change(**args)
    print(*output)

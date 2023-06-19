#!/usr/bin/env python3
# -*- coding: utf-8 -*-
from itertools import combinations
from itertools import permutations
from scipy.stats import levene
from scipy.stats import mannwhitneyu
from scipy.stats import shapiro
from skbio.stats.composition import ancom
from skbio.stats.composition import multiplicative_replacement as m
from statsmodels.stats.multitest import fdrcorrection 
import argparse
import numpy as np
import pandas as pd

def shapiro(df):
    output = pd.DataFrame()
    for col in df.columns: 
        for cat in df.index.unique():
            output.loc[col,cat] = shapiro(df.loc[cat,col])[1]
    return output

def levene(df):
    output = pd.Series()
    for col in df.columns: 
        output[col] = levene(*[df.loc[cat,col] for cat in df.index.unique()])[1]
    return output

def ANCOM(df, perm=False):
    combs = list(combinations(df.index.unique(), 2))
    if perm: combs = list(permutations(df.index.unique(), 2))
    outdf = pd.DataFrame(
            [ancom(pd.concat([df.loc[i[0]], df.loc[i[1]]]), pd.concat([df.loc[i[0]], df.loc[i[1]]]).index.to_series())[0]['Reject null hypothesis'] for i in combs],
            columns = df.columns,
            index = combs,
            )
    return outdf

def sig(df, mult=False, perm=False):
    combs = list(combinations(df.index.unique(), 2))
    if perm: combs = list(permutations(df.index.unique(), 2))
    outdf = pd.DataFrame(
        [mannwhitneyu(df.loc[i[0]], df.loc[i[1]])[1] for i in combs],
        columns = df.columns,
        index = combs
        ).T
    if mult:
        outdf = pd.DataFrame(
            fdrcorrection(outdf.values.flatten())[1].reshape(outdf.shape),
            columns = outdf.columns,
            index = outdf.index
            )
    return outdf

def lfc(df, mult=False, perm=False):
    if mult: df = pd.DataFrame(m(df), index=df.index, columns=df.columns)
    combs = list(combinations(df.index.unique(), 2))
    if perm: combs = list(permutations(df.index.unique(), 2))
    outdf = pd.DataFrame(np.array(
        [df.loc[i[0]].mean().div(df.loc[i[1]].mean()) for i in combs]),
        columns = df.columns,
        index = combs
        ).T.apply(np.log2)
    return outdf

def change(subject, mult=False, perm=False):
    df = pd.read_csv(f'../results/{subject}.tsv', sep='\t', index_col=0)
    fc = lfc(df, mult=mult, perm=perm)
    fc.columns = fc.columns.str.join('/')
    fc = fc.replace([np.inf, -np.inf], np.nan)
    pval = sig(df, mult=mult, perm=perm)
    pval.columns = pval.columns.str.join('/')
    fc = fc.set_axis(['Log2(' + fc.columns[0] + ')'], axis=1)
    pval = pval.set_axis(['MWW_q-value'], axis=1)
    basemean = df.mean().to_frame('basemean')
    means = df.groupby(level=0).mean().T
    means.columns = means.columns + '_Mean'
    baseprevail = df.agg(np.count_nonzero, axis=0).div(df.shape[0]).to_frame('baseprevail')
    prevail = df.groupby(level=0, axis=0).apply(lambda x: x.agg(np.count_nonzero, axis=0).div(x.shape[0])).T
    prevail.columns = prevail.columns + '_Prev'
    output = pd.concat([basemean,means,baseprevail,prevail,fc,pval], join='inner', axis=1).sort_values('MWW_q-value')
    output.to_csv(f'../results/{subject}change.tsv', sep='\t')
    return output

if __name__ == '__main__':
    parser = argparse.ArgumentParser(description='Change - Produces a report of the significant feature changes')
    parser.add_argument('subject')
    parser.add_argument('-m', '--mult')
    parser.add_argument('-p', '--perm')
    args = parser.parse_args()
    args = {k: v for k, v in vars(args).items() if v is not None}
    output = change(**args)
    print(*output)

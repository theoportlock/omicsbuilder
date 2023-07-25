#!/usr/bin/env python
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
import matplotlib.pyplot as plt
import numpy as np
import os
import pandas as pd
import seaborn as sns
import subprocess

def shapiro(subject, **kwargs):
    df = pd.read_csv(f'../results/{subject}.tsv', sep='\t', index_col=0).sort_index()
    output = pd.DataFrame()
    for col in df.columns: 
        for cat in df.index.unique():
            output.loc[col,cat] = shapiro(df.loc[cat,col])[1]
    return output

def levene(subject, **kwargs):
    df = pd.read_csv(f'../results/{subject}.tsv', sep='\t', index_col=0).sort_index()
    output = pd.Series()
    for col in df.columns: 
        output[col] = levene(*[df.loc[cat,col] for cat in df.index.unique()])[1]
    return output

def ANCOM(subject, **kwargs):
    df = pd.read_csv(f'../results/{subject}.tsv', sep='\t', index_col=0).sort_index()
    combs = list(combinations(df.index.unique(), 2))
    if kwargs.get('perm'): combs = list(permutations(df.index.unique(), 2))
    outdf = pd.DataFrame(
            [ancom(pd.concat([df.loc[i[0]], df.loc[i[1]]]), pd.concat([df.loc[i[0]], df.loc[i[1]]]).index.to_series())[0]['Reject null hypothesis'] for i in combs],
            columns = df.columns,
            index = combs,
            )
    return outdf

def LEFSE(subject, **kwargs):
    df = pd.read_csv(f'../results/{subject}.tsv', sep='\t', index_col=0).sort_index()
    ndf = df.T
    ndf.index.name = 'class'
    ndf = ndf.T.reset_index().T
    ndf.to_csv(f'../results/{subject}LEFSE_data.txt', sep='\t')
    #df.to_csv(f'../results/{subject}LEFSE_data.txt', sep='\t')
    #os.system(f'lefse_format_input.py ../results/{subject}LEFSE_data.txt ../results/{subject}LEFSE_format.txt -f r -c 2 -s -1 -u 1 -o 1000000')
    #os.system(f'lefse_format_input.py ../results/{subject}.tsv ../results/{subject}LEFSE_format.txt -f c -u 1')
    rs.system(f'lefse_format_input.py ../results/{subject}LEFSE_data.txt ../results/{subject}LEFSE_format.txt -f r -c 1 -u 1 -o 1000000')
    os.system(f'lefse_run.py ../results/{subject}LEFSE_format.txt ../results/{subject}LEFSE_scores.txt -l 2 --verbose 1')
    outdf = pd.read_csv(f'../results/{subject}LEFSE_scores.txt', sep='\t', index_col=0).T
    return outdf

def mww(subject, **kwargs):
    df = pd.read_csv(f'../results/{subject}.tsv', sep='\t', index_col=0).sort_index()
    df.index, df.columns = df.index.astype(str), df.columns.astype(str)
    combs = list(combinations(df.index.unique(), 2))
    if kwargs.get('perm'): combs = list(permutations(df.index.unique(), 2))
    outdf = pd.DataFrame(
        [mannwhitneyu(df.loc[i[0]], df.loc[i[1]])[1] for i in combs],
        columns = df.columns,
        index = combs
        ).T
    if kwargs.get('mult'):
        outdf = pd.DataFrame(
            fdrcorrection(outdf.values.flatten())[1].reshape(outdf.shape),
            columns = outdf.columns,
            index = outdf.index
            )
    outdf.columns = outdf.columns.str.join('/')
    outdf = outdf.add_prefix('mww_sig(').add_suffix(')')
    outdf = outdf.replace([np.inf, -np.inf], np.nan)
    return outdf

def lfc(subject, **kwargs):
    df = pd.read_csv(f'../results/{subject}.tsv', sep='\t', index_col=0).sort_index()
    df.index, df.columns = df.index.astype(str), df.columns.astype(str)
    combs = list(combinations(df.index.unique(), 2))
    if kwargs.get('perm'): combs = list(permutations(df.index.unique(), 2))
    outdf = pd.DataFrame(np.array(
        [df.loc[i[0]].mean().div(df.loc[i[1]].mean()) for i in combs]),
        columns = df.columns,
        index = combs
        ).T.apply(np.log2)
    outdf.columns = outdf.columns.str.join('/')
    outdf = outdf.add_prefix('log2(').add_suffix(')')
    outdf = outdf.replace([np.inf, -np.inf], np.nan)
    return outdf

def prevail(subject, **kwargs):
    df = pd.read_csv(f'../results/{subject}.tsv', sep='\t', index_col=0).sort_index()
    df.index, df.columns = df.index.astype(str), df.columns.astype(str)
    basemean = df.mean().to_frame('basemean')
    means = df.groupby(level=0).mean().T
    means.columns = means.columns + '_Mean'
    baseprevail = df.agg(np.count_nonzero, axis=0).div(df.shape[0]).to_frame('baseprevail')
    prevail = df.groupby(level=0, axis=0).apply(lambda x: x.agg(np.count_nonzero, axis=0).div(x.shape[0])).T
    prevail.columns = prevail.columns + '_Prev'
    output = pd.concat([basemean,means,baseprevail,prevail], join='inner', axis=1)
    return output

def change(subject, analysis, **kwargs):
    available={
        'prevail':prevail,
        'mww':mww,
        'lfc':lfc,
        'lefse':LEFSE,
        'ancom':ANCOM,
        }
    output = []
    i = analysis[0]
    for i in analysis:
        output.append(available.get(i)(subject, **kwargs))
    out = pd.concat(output, join='inner', axis=1)
    out.to_csv(f'../results/{subject}change.tsv', sep='\t')
    return out

if __name__ == '__main__':
    parser = argparse.ArgumentParser(description='Change - Produces a report of the significant feature changes')
    parser.add_argument('subject')
    parser.add_argument('-a', '--analysis', nargs='+', default=['prevail','mww','lfc'])
    parser.add_argument('--mult', action='store_true')
    args, unknown = parser.parse_known_args()
    args = {k: v for k, v in vars(args).items() if v is not None}
    kwargs = eval(unknown[0]) if unknown != [] else {}
    output = change(**args|kwargs)
    print(output)

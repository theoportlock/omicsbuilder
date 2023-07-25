#!/usr/bin/env python
# -*- coding: utf-8 -*-
import argparse
import pandas as pd

def describe(subject, pval=0.05, corr=None, change=None, sig=None):
    # CHANGED
    df = pd.read_csv(f'../results/{subject}.tsv', sep='\t', index_col=0)
    if change and sig:
        changed = 'sig changed = ' +\
            str(df[sig].lt(pval).sum()) + '/' + str(df.shape[0]) + ' (' + str(round(df[sig].lt(pval).sum()/df.shape[0] * 100)) + '%)'
        # INCREASED
        increased = 'sig increased = ' +\
            str(df.loc[(df[sig].lt(pval)) & (df[change].gt(0)),sig].lt(pval).sum()) +\
            '/' +\
            str(df.shape[0]) +\
            ' (' +\
            str(round(df.loc[(df[sig].lt(pval)) & (df[change].gt(0)),sig].lt(pval).sum()/df.shape[0] * 100)) +\
            '%)'
        # DECREASED
        decreased = 'sig decreased = ' +\
            str(df.loc[(df[sig].lt(pval)) & (df[change].lt(0)),sig].lt(pval).sum()) +\
            '/' +\
            str(df.shape[0]) +\
            ' (' +\
            str(round(df.loc[(df[sig].lt(pval)) & (df[change].lt(0)),sig].lt(pval).sum()/df.shape[0] * 100)) +\
            '%)'
        summary = pd.DataFrame([changed, increased, decreased], columns=[subject])
    else:
        summary = df.describe(include='all').T.reset_index()
    if corr:
        changed = 'sig correlated = ' +\
            str(df['sig'].lt(pval).sum()) + '/' + str(df.shape[0]) + ' (' + str(round(df['sig'].lt(pval).sum()/df.shape[0] * 100)) + '%)'
        # INCREASED
        increased = 'sig positively correlated = ' +\
            str(df.loc[(df['sig'].lt(pval)) & (df[df.columns[df.columns.str.contains('rho')]].gt(0).iloc[:,0]), 'sig'].lt(pval).sum()) +\
            '/' +\
            str(df.shape[0]) +\
            ' (' +\
            str(round(df.loc[(df['sig'].lt(pval)) & (df[df.columns[df.columns.str.contains('rho')]].gt(0).iloc[:,0]), 'sig'].lt(pval).sum()/df.shape[0] * 100)) +\
            '%)'
        # DECREASED
        decreased = 'sig negatively correlated = ' +\
            str(df.loc[(df['sig'].lt(pval)) & (df[df.columns[df.columns.str.contains('rho')]].lt(0).iloc[:,0]), 'sig'].lt(pval).sum()) +\
            '/' +\
            str(df.shape[0]) +\
            ' (' +\
            str(round(df.loc[(df['sig'].lt(pval)) & (df[df.columns[df.columns.str.contains('rho')]].lt(0).iloc[:,0]), 'sig'].lt(pval).sum()/df.shape[0] * 100)) +\
            '%)'
        summary = pd.DataFrame([changed, increased, decreased], columns=[subject])
    summary.to_csv(f'../results/{subject}describe.tsv', sep='\t', index=False)
    return summary

if __name__ == '__main__':
    parser = argparse.ArgumentParser(description='Describe - Produces a summary report of analysis')
    parser.add_argument('subject')
    parser.add_argument('-p', '--pval', type=float)
    parser.add_argument('-c', '--change')
    parser.add_argument('-s', '--sig')
    parser.add_argument('-r', '--corr', action='store_true')
    args = parser.parse_args()
    args = {k: v for k, v in vars(args).items() if v is not None}
    output = describe(**args)
    print(output)

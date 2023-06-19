#!/usr/bin/env python3
# -*- coding: utf-8 -*-
import argparse
import pandas as pd

def describe(subject, pval):
    # CHANGED
    df = pd.read_csv(f'../results/{subject}.tsv', sep='\t', index_col=0)
    changed = 'sig changed = ' +\
        str(df['MWW_q-value'].lt(pval).sum()) + '/' + str(df.shape[0]) + ' (' + str(round(df['MWW_q-value'].lt(pval).sum()/df.shape[0] * 100)) + '%)'
    # INCREASED
    increased = 'sig increased = ' +\
        str(df.loc[(df['MWW_q-value'].lt(pval)) & (df[df.columns[df.columns.str.contains('Log2')]].gt(0).iloc[:,0]), 'MWW_q-value'].lt(pval).sum()) +\
        '/' +\
        str(df.shape[0]) +\
        ' (' +\
        str(round(df.loc[(df['MWW_q-value'].lt(pval)) & (df[df.columns[df.columns.str.contains('Log2')]].gt(0).iloc[:,0]), 'MWW_q-value'].lt(pval).sum()/df.shape[0] * 100)) +\
        '%)'
    # DECREASED
    decreased = 'sig decreased = ' +\
        str(df.loc[(df['MWW_q-value'].lt(pval)) & (df[df.columns[df.columns.str.contains('Log2')]].lt(0).iloc[:,0]), 'MWW_q-value'].lt(pval).sum()) +\
        '/' +\
        str(df.shape[0]) +\
        ' (' +\
        str(round(df.loc[(df['MWW_q-value'].lt(pval)) & (df[df.columns[df.columns.str.contains('Log2')]].lt(0).iloc[:,0]), 'MWW_q-value'].lt(pval).sum()/df.shape[0] * 100)) +\
        '%)'
    summary = pd.DataFrame([changed, increased, decreased], columns=[subject])
    summary.to_csv(f'../results/{subject}describe.tsv', sep='\t', index=False)
    return summary

if __name__ == '__main__':
    parser = argparse.ArgumentParser(description='Describe - Produces a report of the significant feature changes')
    parser.add_argument('subject')
    parser.add_argument('-p', '--pvalthresh', type=float)
    args = parser.parse_args()
    args = {k: v for k, v in vars(args).items() if v is not None}
    output = describe(**args)
    print(*output)

#!/usr/bin/env python
# -*- coding: utf-8 -*-

from scipy.stats import spearmanr
from statsmodels.stats.multitest import fdrcorrection
import argparse
import pandas as pd

def corr(subject, mult=True):
    def to_edges(df):
        df = df.rename_axis('source', axis=0).rename_axis("target", axis=1)
        edges = df.stack().to_frame()[0]
        nedges = edges.reset_index()
        edges = nedges[nedges.target != nedges.source].set_index(['source','target'])[0]
        return edges
    df = pd.read_csv(f'../results/{subject}.tsv', sep='\t', index_col=0)
    cor, pval = spearmanr(df)
    cordf = pd.DataFrame(cor, index=df.columns, columns=df.columns)
    pvaldf = pd.DataFrame(pval, index=df.columns, columns=df.columns)
    if mult:
        pvaldf = pd.DataFrame(
            fdrcorrection(pvaldf.values.flatten())[1].reshape(pvaldf.shape),
            index = pvaldf.index,
            columns = pvaldf.columns)
    outdf = to_edges(cordf).to_frame('rho').join(to_edges(pvaldf).to_frame('sig')).sort_values('rho')
    outdf.to_csv(f'../results/{subject}corr.tsv', sep='\t')
    return outdf


if __name__ == '__main__':
    parser = argparse.ArgumentParser(description='Corr - Produces a report of the significant correlations between data')
    parser.add_argument('subject')
    parser.add_argument('-m', '--mult', action='store_true')
    args = parser.parse_args()
    args = {k: v for k, v in vars(args).items() if v is not None}
    output = corr(**args)
    print(output)

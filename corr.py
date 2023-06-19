#!/usr/bin/env python3
# -*- coding: utf-8 -*-

from networkx.algorithms.community.centrality import girvan_newman as cluster
from scipy.stats import spearmanr
from statsmodels.stats.multitest import fdrcorrection
import argparse
import pandas as pd

def to_edges(df, thresh=0.5, directional=True):
    df = df.rename_axis('source', axis=0).rename_axis("target", axis=1)
    edges = df.stack().to_frame()[0]
    nedges = edges.reset_index()
    edges = nedges[nedges.target != nedges.source].set_index(['source','target']).drop_duplicates()[0]
    if directional:
        fin = edges.loc[(edges < 0.99) & (edges.abs() > thresh)].dropna().reset_index().rename(columns={'level_0': 'source', 'level_1':'target', 0:'weight'}).set_index(['source','target']).sort_values('weight')
    else:
        fin = edges.loc[(edges < 0.99) & (edges > thresh)].dropna().reset_index().rename(columns={'level_0': 'source', 'level_1':'target', 0:'weight'}).set_index(['source','target']).sort_values('weight')
    return fin
# STOPPED HERE

def cluster(G):
    communities = cluster(G)
    node_groups = []
    for com in next(communities):
        node_groups.append(list(com))
    df = pd.DataFrame(index=G.nodes, columns=['Group'])
    for i in range(pd.DataFrame(node_groups).shape[0]):
        tmp = pd.DataFrame(node_groups).T[i].to_frame().dropna()
        df.loc[tmp[i], 'Group'] = i
    return df.Group

def corr(df1, df2, mult=True):
    if subject:
        df = pd.read_csv(f'../results/{subject}.tsv', sep='\t', index_col=0)
    else:
        df1 = pd.read_csv(f'../results/{df1}.tsv', sep='\t', index_col=0)
        df2 = pd.read_csv(f'../results/{df2}.tsv', sep='\t', index_col=0)
        df = df1.join(df2, how='inner')
    cor, pval = spearmanr(df)
    cordf = pd.DataFrame(cor, index=df.columns, columns=df.columns)
    pvaldf = pd.DataFrame(pval, index=df.columns, columns=df.columns)
    if not subject:
        cordf = cordf.loc[df1.columns, df2.columns]
        pvaldf = pvaldf.loc[df1.columns, df2.columns]
    #pvaldf.fillna(1, inplace=True)
    if mult:
        pvaldf = pd.DataFrame(
            fdrcorrection(pvaldf.values.flatten())[1].reshape(pvaldf.shape),
            index = pvaldf.index,
            columns = pvaldf.columns)
    outdf = to_edges(cordf).join(to_edges(pvaldf))
    return cordf, pvaldf


if __name__ == '__main__':
    parser = argparse.ArgumentParser(description='Corr - Produces a report of the significant correlations between data')
    parser.add_argument('subject')
    parser.add_argument('-df1')
    parser.add_argument('-df2')
    parser.add_argument('-m', '--mult')
    args = parser.parse_args()
    args = {k: v for k, v in vars(args).items() if v is not None}
    output = corr(**args)
    print(*output)

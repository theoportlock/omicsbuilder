#!/usr/bin/env python
# -*- coding: utf-8 -*-

from scipy.spatial import distance
from skbio.stats.distance import permanova
import argparse
import numpy as np
import pandas as pd
import skbio

def PERMANOVA(df, pval=True, full=False):
    Ar_dist = distance.squareform(distance.pdist(df, metric="braycurtis"))
    DM_dist = skbio.stats.distance.DistanceMatrix(Ar_dist)
    result = permanova(DM_dist, df.index)
    if full:
        return result
    if pval:
        return result['p-value']
    else:
        return result['test statistic']

def explainedvariance(df1, df2, pval=True):
    # how does df1 explain variance in df2 where df2 is meta (only categories)
    # should rework this one to include in calculate but hard
    # 4.32 is significant
    DF1 = pd.read_csv(f'../results/{df1}.tsv', sep='\t', index_col=0).dropna()
    DF2 = pd.read_csv(f'../results/{df2}.tsv', sep='\t', index_col=0).dropna()
    target = DF2.columns[0]
    output = pd.Series()
    for target in DF2.columns:
        tdf = DF1.join(DF2[target].fillna('missing'),how='inner').set_index(target)
        if all(tdf.index.value_counts().lt(10)):
            continue
        if tdf.index.nunique() <= 1:
            continue
        output[target] = PERMANOVA(tdf, pval=pval)
    if pval:
        power = -output.apply(np.log2)
    else:
        power = output
    power = power.to_frame(df1)
    power.to_csv(f'../results/{df1}{df2}power.tsv', sep='\t')
    return power

def variance(df1):
    DF1 = pd.read_csv(f'../results/{df1}.tsv', sep='\t', index_col=0).dropna()
    output = PERMANOVA(DF1, full=True)
    with open(f'../results/{df1}variance.txt','w') as of: of.write(output.to_string())
    return output

if __name__ == '__main__':
    parser = argparse.ArgumentParser(description='Calculates the explained variance of the categorical labels of the metadata')
    parser.add_argument('-df1', required=True)
    parser.add_argument('-df2', required=False)
    parser.add_argument('-p', '--pval')
    args = parser.parse_args()
    args = {k: v for k, v in vars(args).items() if v is not None}
    if args.get('df2'):
        output = explainedvariance(**args)
    else:
        output = variance(**args)
    print(output)

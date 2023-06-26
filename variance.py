#!/usr/bin/env python
# -*- coding: utf-8 -*-

from scipy.spatial import distance
from skbio.stats.distance import permanova
import argparse
import numpy as np
import pandas as pd
import skbio

def variance(df1, df2, pval=True):
    # how does df1 explain variance in df2 where df2 is meta (only categories)
    # should rework this one to include in calculate but hard
    def varianceexplained(df, pval=True):
        Ar_dist = distance.squareform(distance.pdist(df, metric="braycurtis"))
        DM_dist = skbio.stats.distance.DistanceMatrix(Ar_dist)
        result = permanova(DM_dist, df.index)
        if pval:
            return result['p-value']
        else:
            return result['test statistic']
    DF1 = pd.read_csv(f'../results/{df1}.tsv', sep='\t', index_col=0)
    DF2 = pd.read_csv(f'../results/{df2}.tsv', sep='\t', index_col=0)
    target = DF2.columns[0]
    output = pd.Series()
    for target in DF2.columns:
        tdf = DF1.join(DF2[target].fillna('missing')).set_index(target)
        if all(tdf.index.value_counts().lt(20)):
            next
        tdf = tdf.loc[tdf.index.value_counts().gt(10)]
        print(tdf.index.value_counts())
        output[target] = f.varianceexplained(tdf, pval=False)
    if pval:
        power = -output.apply(np.log)
    else:
        power = output
    power.to_csv(f'../results/{subject}power.csv')
    return power

if __name__ == '__main__':
    parser = argparse.ArgumentParser(description='Variance - Calculates the explained variance of the categorical labels of the metadata')
    parser.add_argument('subject')
    parser.add_argument('-p', '--pval')
    args = parser.parse_args()
    args = {k: v for k, v in vars(args).items() if v is not None}
    output = variance(**args)
    print(*output)

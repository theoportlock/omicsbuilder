#!/usr/bin/env python3
# -*- coding: utf-8 -*-
from scipy.spatial import distance
from skbio import DistanceMatrix
from skbio.diversity.alpha import pielou_e
from skbio.diversity.alpha import shannon
from skbio.stats.distance import permanova
import argparse
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd

def PERMANOVA(df, meta):
    beta = pd.DataFrame(
        distance.squareform(distance.pdist(df, metric="braycurtis")),
        columns=df.index,
        index=df.index)
    pvals = permanova(DistanceMatrix(beta, beta.index), meta)
    return pvals

def Richness(df, axis=1):
    return df.agg(np.count_nonzero, axis=axis)

def Evenness(df, axis=1):
    return df.agg(pielou_e, axis=axis)

def Shannon(df, axis=1):
    return df.agg(shannon, axis=axis)

def diversity(subject):
    df = pd.read_csv(f'../results/{subject}.tsv', sep='\t', index_col=0)
    diversity = pd.concat(
            [Evenness(df).to_frame('Evenness'),
             Richness(df).to_frame('Richness'),
             Shannon(df).to_frame('Shannon')],
            axis=1).sort_index().sort_index(ascending=False)
    diversity.to_csv(f'../results/{subject}diversity.tsv', sep='\t')
    return diversity

if __name__ == '__main__':
    parser = argparse.ArgumentParser(description='diversity - Produces a report of diversity measures of a dataset')
    parser.add_argument('subject')
    args = parser.parse_args()
    args = {k: v for k, v in vars(args).items() if v is not None}
    output = diversity(**args)
    print(output)

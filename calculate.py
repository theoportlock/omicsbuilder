#!/usr/bin/env python
# -*- coding: utf-8 -*-

from scipy.spatial import distance
from skbio.diversity.alpha import pielou_e
from skbio.diversity.alpha import shannon
from sklearn.decomposition import PCA
from sklearn.manifold import MDS
from sklearn.manifold import TSNE
from sklearn.preprocessing import StandardScaler
from sklearn_som.som import SOM
import argparse
import numpy as np
import pandas as pd
import skbio
import sys
import umap

def diversity(df):
    def Richness(df, axis=1): return df.agg(np.count_nonzero, axis=axis)
    def Evenness(df, axis=1): return df.agg(pielou_e, axis=axis)
    def Shannon(df, axis=1): return df.agg(shannon, axis=axis)
    diversity = pd.concat(
            [Evenness(df).to_frame('Evenness'),
             Richness(df).to_frame('Richness'),
             Shannon(df).to_frame('Shannon')],
            axis=1).sort_index().sort_index(ascending=False)
    return diversity

def fbratio(df):
    phylum = df.copy()
    phylum.columns = phylum.columns.str.extract('p__(.*)\|c')[0]
    taxoMsp = phylum.T.groupby(phylum.columns).sum()
    taxoMsp = taxoMsp.loc[taxoMsp.sum(axis=1) != 0, taxoMsp.sum(axis=0) != 0]
    taxoMsp = taxoMsp.T.div(taxoMsp.sum(axis=0), axis=0)
    FB = taxoMsp.Firmicutes.div(taxoMsp.Bacteroidetes)
    FB.replace([np.inf, -np.inf], np.nan, inplace=True)
    FB.dropna(inplace=True)
    FB = FB.reset_index().set_axis(['Host Phenotype', 'FB_Ratio'], axis=1).set_index('Host Phenotype')
    return FB

def pbratio(df):
    genus = df.copy()
    genus.columns = genus.columns.str.extract('g__(.*)\|s')[0]
    taxoMsp = genus.T.groupby(genus.columns).sum()
    taxoMsp = taxoMsp.loc[taxoMsp.sum(axis=1) != 0, taxoMsp.sum(axis=0) != 0]
    taxoMsp = taxoMsp.T.div(taxoMsp.sum(axis=0), axis=0)
    pb = taxoMsp.Prevotella.div(taxoMsp.Bacteroides)
    pb.replace([np.inf, -np.inf], np.nan, inplace=True)
    pb.dropna(inplace=True)
    pb = pb.reset_index().set_axis(['Host Phenotype', 'PB_Ratio'], axis=1).set_index('Host Phenotype')
    return pb

def pca(df):
    scaledDf = StandardScaler().fit_transform(df)
    pca = PCA()
    results = pca.fit_transform(scaledDf).T
    df['PC1'], df['PC2'] = results[0,:], results[1,:]
    return df[['PC1', 'PC2']]

def pcoa(df):
    Ar_dist = distance.squareform(distance.pdist(df, metric="braycurtis"))
    DM_dist = skbio.stats.distance.DistanceMatrix(Ar_dist)
    PCoA = skbio.stats.ordination.pcoa(DM_dist, number_of_dimensions=2)
    results = PCoA.samples.copy()
    df['PC1'], df['PC2'] = results.iloc[:,0].values, results.iloc[:,1].values
    return df[['PC1', 'PC2']]

def nmds(df):
    BC_dist = pd.DataFrame(
        distance.squareform(distance.pdist(df, metric="braycurtis")),
        columns=df.index,
        index=df.index) 
    mds = MDS(n_components = 2, metric = False, max_iter = 500, eps = 1e-12, dissimilarity = 'precomputed')
    results = mds.fit_transform(BC_dist)
    df['PC1'], df['PC2'] = results[:,0], results[:,1]
    return df[['PC1', 'PC2']]

def tsne(df):
    results = TSNE(n_components=2, learning_rate='auto', init='random').fit_transform(df)
    df['PC1'], df['PC2'] = results[:,0], results[:,1]
    return df[['PC1', 'PC2']]

def top20(df, **kwargs):
    if df.shape[1] > 20:
        df['other'] = df[df.sum().sort_values(ascending=False).iloc[19:].index].sum(axis=1)
    df = df[df.sum().sort_values().tail(20).index]
    return df

def som(df):
    som = SOM(m=3, n=1, dim=2)
    som.fit(df)
    return som

def umap(df):
    scaledDf = StandardScaler().fit_transform(df)
    reducer = umap.UMAP()
    results = reducer.fit_transform(scaledDf)
    df['PC1'], df['PC2'] = results[:,0], results[:,1]
    return df[['PC1', 'PC2']]

def calculate(analysis, subject, **kwargs):
    df = pd.read_csv(f'../results/{subject}.tsv', sep='\t', index_col=0)
    available={
        'diversity':diversity,
        'fbratio':fbratio,
        'pbratio':pbratio,
        'pca':pca,
        'pcoa':pcoa,
        'nmds':nmds,
        'tsne':tsne,
        'top20':top20,
        'som':som,
        'umap':umap,
        }
    output = available.get(analysis)(df, **kwargs)
    output.to_csv(f'../results/{subject}{analysis}.tsv', sep='\t')
    return output

if __name__ == '__main__':
    parser = argparse.ArgumentParser(description='Calculate - compute a value for each sample based on features')
    parser.add_argument('analysis')
    parser.add_argument('subject')
    args, unknown = parser.parse_known_args()
    args = {k: v for k, v in vars(args).items() if v is not None}
    kwargs = eval(unknown[0]) if unknown != [] else {}
    output = calculate(**args|kwargs)
    print(output)

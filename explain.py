#!/usr/bin/env python
# -*- coding: utf-8 -*-

from scipy.stats import spearmanr
import argparse
import pickle
import numpy as np
import pandas as pd
import shap

def SHAP_interact(subject):
    df = pd.read_csv(f'../results/{subject}.tsv', sep='\t', index_col=0)
    X = df.copy()
    with open(f'../results/{subject}predict.pkl', 'rb') as file: model = pickle.load(file)
    explainer = shap.TreeExplainer(model)
    inter_shaps_values = explainer.shap_interaction_values(X)
    vals = inter_shaps_values[0]
    for i in range(1, vals.shape[0]):
        vals[0] += vals[i]
    final = pd.DataFrame(vals[0], index=X.columns, columns=X.columns)
    final = final.stack().sort_values().to_frame('SHAP_interaction')
    final.index = final.index.set_names(['source', 'target'], level=[0,1])
    final.to_csv(f'../results/{subject}shapinteract.tsv', sep='\t')
    return final

def SHAP_bin(subject):
    df = pd.read_csv(f'../results/{subject}.tsv', sep='\t', index_col=0)
    X = df.copy()
    with open(f'../results/{subject}predict.pkl', 'rb') as file: model = pickle.load(file)
    explainer = shap.TreeExplainer(model)
    shaps_values = explainer(X)
    meanabsshap = pd.Series(
        np.abs(shaps_values.values[:, :, 0]).mean(axis=0),
        index=X.columns
    )
    corrs = [spearmanr(shaps_values.values[:, x, 1], X.iloc[:, x])[0] for x in range(len(X.columns))]
    final = meanabsshap * np.sign(corrs)
    final.fillna(0, inplace=True)
    final = final.sort_values()
    final.to_frame('Shap_Value').to_csv(f'../results/{subject}shaps.tsv', sep='\t')
    return final

def explain(analysis, subject, **kwargs):
    available={
        'SHAP_bin':SHAP_bin,
        'SHAP_interact':SHAP_interact
        }
    df = available.get(analysis)(subject, **kwargs)
    return df

if __name__ == '__main__':
    parser = argparse.ArgumentParser(description='Explain - compute a SHAP value for each sample based on features in AI model')
    parser.add_argument('analysis')
    parser.add_argument('subject')
    args, unknown = parser.parse_known_args()
    args = {k: v for k, v in vars(args).items() if v is not None}
    kwargs = eval(unknown[0]) if unknown != [] else {}
    output = explain(**args|kwargs)
    print(output)

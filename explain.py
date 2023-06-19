#!/usr/bin/env python3
# -*- coding: utf-8 -*-

from scipy.stats import spearmanr
import argparse
import numpy as np
import pandas as pd
import shap

def SHAP_interact(X, model):
    explainer = shap.TreeExplainer(model)
    inter_shaps_values = explainer.shap_interaction_values(X)
    vals = inter_shaps_values[0]
    for i in range(1, vals.shape[0]):
        vals[0] += vals[i]
    final = pd.DataFrame(vals[0], index=X.columns, columns=X.columns)
    return final

def SHAP_bin(X, model):
    explainer = shap.TreeExplainer(model)
    shaps_values = explainer(X)
    meanabsshap = pd.Series(
        np.abs(shaps_values.values[:, :, 0]).mean(axis=0),
        index=X.columns
    )
    corrs = [spearmanr(shaps_values.values[:, x, 1], X.iloc[:, x])[0] for x in range(len(X.columns))]
    final = meanabsshap * np.sign(corrs)
    final.fillna(0, inplace=True)
    return final

def explain(X,model):
    return None

if __name__ == '__main__':
    parser = argparse.ArgumentParser(description='Explain - Explains the machine learning models')
    parser.add_argument('subject')
    parser.add_argument('-m', '--mult')
    parser.add_argument('-p', '--perm')
    args = parser.parse_args()
    args = {k: v for k, v in vars(args).items() if v is not None}
    output = explain(**args)
    print(*output)

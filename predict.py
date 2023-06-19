#!/usr/bin/env python3
# -*- coding: utf-8 -*-
from itertools import combinations
from sklearn.ensemble import RandomForestClassifier
from sklearn.ensemble import RandomForestRegressor
from sklearn.metrics import auc
from sklearn.metrics import classification_report
from sklearn.metrics import confusion_matrix
from sklearn.metrics import mean_absolute_error, r2_score
from sklearn.metrics import roc_auc_score
from sklearn.metrics import roc_curve
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
import argparse
import functions as f
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd

def classifier(X, y):
    model = RandomForestClassifier(n_estimators=500, n_jobs=-1, random_state=1)
    X_train, X_test, y_train, y_test = train_test_split(X, y, random_state = 1, stratify=y)
    model.fit(X_train, y_train)
    y_pred = model.predict(X_test)
    print(classification_report(y_true=y_test, y_pred=y_pred))
    print('AUCROC =', roc_auc_score(y, model.predict_proba(X)[:, 1]))
    print(confusion_matrix(y_test, y_pred))
    return model

def regressor(df, subject):
    tdf = f.StandardScale(df.copy())
    model = RandomForestRegressor(n_estimators=500, n_jobs=-1, random_state=1)
    X, y = tdf, tdf.index
    X_train, X_test, y_train, y_test = train_test_split(tdf, tdf.index, random_state = 1)
    model.fit(X_train, y_train)
    y_pred = model.predict(X_test)
    mae = mean_absolute_error(y_test, y_pred)
    r2 = r2_score(y_test, y_pred)
    print("Mean Absolute Error:", mae)
    print("R-squared score:", r2)
    #plot actual vs pred
    plt.scatter(y_test, y_pred, alpha=0.5, color='darkblue', marker='o')
    plt.plot([min(y_test), max(y_test)], [min(y_test), max(y_test)], '--', color='gray', linewidth=2)
    for i in range(len(y_test)):
        plt.plot([y_test[i], y_test[i]], [y_test[i], y_pred[i]], color='red', alpha=0.5)
    plt.xlabel('Actual Values')
    plt.ylabel('Predicted Values')
    plt.title('Random Forest Regression Model')
    plt.savefig('../results/expvsobs.svg')
    plt.show()
    # SHAP analysis
    explainer = shap.Explainer(model, X_train)
    shap_values = explainer(X_test)
    shap.plots.waterfall(shap_values[1])
    plt.savefig('../results/waterfall.svg')
    meanabsshap = pd.Series(
            [shap_values.values[:,i].mean() for i in range(shap_values.values.shape[1])],
            index=X.columns
            )
    corrs = [spearmanr(shap_values.values[:,x], X_test.iloc[:,x])[0] for x in range(len(X.columns))]
    final = meanabsshap * np.sign(corrs)
    final.fillna(0, inplace=True)
    final.to_csv(f'../results/{subject}SHAP.csv')

if __name__ == '__main__':
    parser = argparse.ArgumentParser(description='Predict - Produces models that predict the dataframe index')
    parser.add_argument('subject')
    parser.add_argument('-m', '--model')
    args = parser.parse_args()
    args = {k: v for k, v in vars(args).items() if v is not None}
    output = change(**args)
    print(*output)

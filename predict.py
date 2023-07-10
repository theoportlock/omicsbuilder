#!/usr/bin/env python
# -*- coding: utf-8 -*-

from sklearn.ensemble import RandomForestClassifier
from sklearn.ensemble import RandomForestRegressor
from sklearn.metrics import classification_report
from sklearn.metrics import confusion_matrix
from sklearn.metrics import mean_absolute_error, r2_score
from sklearn.metrics import roc_auc_score
from sklearn.metrics import roc_curve
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
import argparse
import pandas as pd
import pickle

def classifier(subject):
    tdf = pd.read_csv(f'../results/{subject}.tsv', sep='\t', index_col=0)
    model = RandomForestClassifier(n_jobs=-1, random_state=1, oob_score=True)
    X, y = tdf, tdf.index
    X_train, X_test, y_train, y_test = train_test_split(X, y, random_state=1)
    model.fit(X_train, y_train)
    y_pred = model.predict(X_test)
    performance = classification_report(y_true=y_test, y_pred=y_pred) + '\n' \
        'AUCROC=' + str(roc_auc_score(y_test, model.predict_proba(X_test)[:, 1])) + '\n\n' +\
        pd.DataFrame(confusion_matrix(y_test, y_pred)).to_string() + '\n' +\
        'oob score=' + str(model.oob_score_)
    print(performance)
    with open(f'../results/{subject}performance.txt', 'w') as of: of.write(performance)
    return model

def regressor(subject):
    tdf = pd.read_csv(f'../results/{subject}.tsv', sep='\t', index_col=0)
    model = RandomForestRegressor(n_jobs=-1, random_state=1, oob_score=True)
    X, y = tdf, tdf.index
    X_train, X_test, y_train, y_test = train_test_split(X, y, random_state=1)
    model.fit(X_train, y_train)
    y_pred = model.predict(X_test)
    mae = mean_absolute_error(y_test, y_pred)
    r2 = r2_score(y_test, y_pred)
    performance = "Mean Absolute Error:" + str(mae) + '\n\n' + \
        "R-squared score:" + str(r2) + '\n' + \
        'oob score =' + str(model.oob_score_)
    print(performance)
    with open(f'../results/{subject}performance.txt', 'w') as of: of.write(performance)
    return model

def predict(analysis, subject, **kwargs):
    available={
        'regressor':regressor,
        'classifier':classifier,
        }
    model = available.get(analysis)(subject, **kwargs)
    with open(f'../results/{subject}predict.pkl', 'wb') as file: pickle.dump(model, file)  
    return model

if __name__ == '__main__':
    parser = argparse.ArgumentParser(description='Calculate - compute a value for each sample based on features')
    parser.add_argument('analysis')
    parser.add_argument('subject')
    args, unknown = parser.parse_known_args()
    args = {k: v for k, v in vars(args).items() if v is not None}
    kwargs = eval(unknown[0]) if unknown != [] else {}
    output = predict(**args|kwargs)
    print(output)

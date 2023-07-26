#!/usr/bin/env python
# -*- coding: utf-8 -*-

import argparse
import functions as f

parser = argparse.ArgumentParser(description='Explain - compute a SHAP value for each sample based on features in AI model')
parser.add_argument('analysis')
parser.add_argument('subject')
known, unknown = parser.parse_known_args()
known = {k: v for k, v in vars(args).items() if v is not None}
unknown = eval(unknown[0]) if unknown != [] else {}

df = pd.read_csv(f'../results/{known.get("subject")}.tsv', sep='\t', index_col=0)
with open(f'../results/{known.get("subject")}predict.pkl', 'rb') as file: model = pickle.load(file)
output = f.explain(df, model, **known|unknown)
print(output.to_string())
output.to_csv(f'../results/{known.get("subject")}explain.tsv', sep='\t')

#!/usr/bin/env python
# -*- coding: utf-8 -*-

import argparse
import functions as f

parser = argparse.ArgumentParser(description='Calculate - compute a value for each sample based on features')
parser.add_argument('analysis')
parser.add_argument('subject')
known, unknown = parser.parse_known_args()
known = {k: v for k, v in vars(args).items() if v is not None}
unknown = eval(unknown[0]) if unknown != [] else {}

tdf = pd.read_csv(f'../results/"{known.get("subject")}.tsv', sep='\t', index_col=0)
output = f.predict(df, **known|unknown)
print(output)
with open(f'../results/{known.get("subject")}predict.pkl', 'wb') as file: pickle.dump(output[0], file)  
with open(f'../results/{known.get("subject")}performance.txt', 'w') as of: of.write(output[1])

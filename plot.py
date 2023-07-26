#!/usr/bin/env python
# -*- coding: utf-8 -*-

import argparse
import functions as f
import matplotlib.pyplot as plt

parser = argparse.ArgumentParser(description='Plot - Produces a plot of a given dataset')
parser.add_argument('plottype')
parser.add_argument('subject')
parser.add_argument('--logx', action='store_true')
parser.add_argument('--logy', action='store_true')
known, unknown = parser.parse_known_args()
known = {k: v for k, v in vars(args).items() if v is not None}
unknown = eval(unknown[0]) if unknown != [] else {}

f.setupplot()
df = pd.read_csv(f'../results/{known.get("subject")}.tsv', sep='\t', index_col=0)
output = plot(df, **known|unknown)
plt.savefig(f'../results/{known.get("subject")}{known.get("plottype")}.svg')
plt.show()

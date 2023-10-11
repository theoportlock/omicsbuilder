#!/usr/bin/env python
# -*- coding: utf-8 -*-

import argparse
import functions as f
import matplotlib.pyplot as plt
import pandas as pd

parser = argparse.ArgumentParser(description='''
Polar - Produces a Polarplot of a given dataset
''')
parser.add_argument('subject')
known, unknown = parser.parse_known_args()
known = {k: v for k, v in vars(known).items() if v is not None}
unknown = eval(unknown[0]) if unknown != [] else {}

f.setupplot()
subject = known.get("subject")
df = f.load(subject)
output = f.polar(df, **unknown)
plt.savefig(f'../results/{subject}polar.svg')
plt.show()

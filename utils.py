#!/usr/bin/env python3
# -*- coding: utf-8 -*-
import pandas as pd
from glob import glob

def reader(name):
    files_found = glob(f'../results/{name}.*')
    if len(files_found) != 1:
        raise ValueError(f"Incorrect number of files found: {len(files_found)}")
    file = files_found[0]
    file_extension = file.split('.')[-1].lower()
    if file_extension in ['csv', 'tsv']:
        delimiter = ',' if file_extension == 'csv' else '\t'
        df = pd.read_csv(file, delimiter=delimiter, index_col=0)
    elif file_extension == 'xlsx':
        df = pd.read_excel(file, index_col=0)
    else:
        raise ValueError(f"Unsupported file extension: {file_extension}")
    return df

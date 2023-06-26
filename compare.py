#!/usr/bin/env python
# -*- coding: utf-8 -*-
import matplotlib.pyplot as plt
import argparse
import sys
import utils


if __name__ == '__main__':
    args, kwargs = utils.reader() 
    main(*args, **kwargs)

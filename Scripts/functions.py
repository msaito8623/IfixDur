# import copy
# import numpy as np
# import os
import pandas as pd
import pyclx.pyclx as px
# import pyldl.mapping as lmap
# import pyldl.measures as lmea
import re
import time
# import xarray as xr
# from multiprocessing import Pool
from pathlib import Path

rdir = './'
idir = '{}/RawData'.format(rdir)
odir = '{}/ProcessedData'.format(rdir)

def measure_time (func):
    def wrapper(*args, **kwargs):
        st = time.time()
        rtn = func(*args, **kwargs)
        ed = time.time()
        print('{:s}: {:4.2f} sec'.format(func.__name__, ed-st))
        return rtn
    return wrapper

@measure_time
def read_clx (name='gml', cols=None, idir=idir):
    clx = px.read_celex(name, idir, usecols=cols)
    clx = rm_dup_cols_celex(clx)
    return clx

def rm_dup_cols_celex (clx):
    clx = clx.loc[:,(~pd.Series(clx.columns).str.contains('[1-9]$', regex=True)).to_list()]
    clx.columns = pd.Series(clx.columns).str.replace('0$', '', regex=True)
    return clx




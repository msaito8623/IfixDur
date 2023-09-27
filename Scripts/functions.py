# import copy
# import numpy as np
# import os
import pandas as pd
# import pyldl.mapping as lmap
# import pyldl.measures as lmea
import re
import time
# import xarray as xr
# from multiprocessing import Pool
from pathlib import Path

rdir = '.'
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
def read_clx (name='gml', cols=None, squeeze=False, nrows=None, idir=idir):
    clx = '{}/{}.cd'.format(idir, name)
    if nrows is None:
        with open(clx, 'r') as f:
            clx = f.readlines()
    else:
        with open(clx, 'r') as f:
            clx = [ next(f) for _ in range(nrows) ]
    clx = pd.Series(clx).str.strip()
    clx = clx.str.split('\\', regex=False, expand=True).fillna('')

    readme_path = '{}/{}.readme'.format(idir, name)
    clx.columns = find_colnames(readme_path, clen=clx.shape[1])

    if not (cols is None):
        colpos = pd.Series(clx.columns).str.replace('[0-9]$','',regex=True).isin(pd.Series(cols))
        clx = clx.loc[:,colpos.to_list()]

    if set([ i[-1] for i in clx.columns if i[-1].isdigit() ])=={'0'}:
        clx.columns = pd.Series(clx.columns).str.replace('0$','',regex=True)
    if squeeze:
        clx = clx.squeeze('columns')
    clx = clx.apply(pd.to_numeric, errors='ignore')
    clx = rm_dup_cols_celex(clx)
    return clx

def find_colnames (readme_path, clen=None):
    with open(readme_path, 'r') as f:
        rd = f.readlines()
    rd = pd.Series(rd).str.strip()
    rd = rd.loc[rd.str.contains('^[0-9]{1,2}\\. +[A-Za-z]+$', regex=True)].reset_index(drop=True)
    rd = rd.str.replace(' +', ' ', regex=True)
    rd = rd.str.split('. ', regex=False, expand=True).rename(columns={0:'ID',1:'Name'})
    cr = (rd.ID.astype(int)+1).iloc[:-1]
    nx = rd.ID.astype(int).shift(-1).iloc[:-1].astype(int)
    if (cr!=nx).any():
        rd = rd.iloc[:cr.index[cr!=nx][0]+1,:]
    assert (( (rd.ID.astype(int)+1) == rd.ID.astype(int).shift(-1)).iloc[:-1]).all()
    rd = rd.Name

    hd = rd.loc[~rd.duplicated(keep=False)].copy().to_list()
    dp = rd.loc[ rd.duplicated(keep='last')].copy().to_list()
    if len(dp)==0:
        assert clen==len(hd)
        rd = hd
    else:
        assert clen%len(dp)==len(hd)
        dp = [dp]*(clen//len(dp))
        dp = [ [ j+str(ind) for j in i ] for ind,i in enumerate(dp) ]
        dp = [ j for i in dp for j in i ]
        rd = hd + dp
    assert clen==len(rd)
    return rd[:clen]

def rm_dup_cols_celex (clx):
    clx = clx.loc[:,(~pd.Series(clx.columns).str.contains('[1-9]$', regex=True)).to_list()]
    clx.columns = pd.Series(clx.columns).str.replace('0$', '', regex=True)
    return clx




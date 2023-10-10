from Scripts.functions import *
import numpy as np
import pandas as pd

gml = init_gml()
gml = add_WordPoS(gml)
gml = add_Mor(gml)
gml = add_MorOrtho(gml)
gml = add_MorPoS(gml)
gml = fix_ambiguous_interfixes(gml)
gml = add_MorCat(gml)
gml = fix_MorCat(gml)
gml = add_LeftOrtho_IfixOrtho_RightOrtho_forNullInterfixCompounds(gml)
gml = add_IsCompound(gml)
gml = add_IfixInd(gml)
gml = add_IfixOrtho(gml)
gml = add_IfixPhono(gml)
gml = add_LeftOrtho(gml)
gml = add_RightOrtho(gml)
gml = limit_to_compounds(gml)
gml = add_inflected_forms(gml)
gml = add_ParaTypeInfl(gml)
gml = add_ParaTypeLemma(gml)
gml = add_ParaTokenInfl(gml)
gml = add_ParaTokenLemma(gml)
gml = add_LeftEntInfl(gml)
gml = add_LeftEntLemma(gml)
gml = add_RightEntInfl(gml)
gml = add_RightEntLemma(gml)



### BELOW IN PROGRESS ###


### Phono ###
gpw = init_gpw()
gml = add_WordPhono(gml, gpw)
gml = add_LeftPhono(gml, gpw)
gml = add_RightPhono(gml, gpw)

def infer_LeftPhono (x):
    if x.LeftPhono=='NotFound':
        res = re.sub('^(.+)' + re.escape(x.IfixPhono + x.RightPhono) + '$', '\\1', x.WordPhono)
    else:
        res = x.LeftPhono
    return res
gml['LeftPhono'] = gml.apply(infer_LeftPhono, axis=1)

def infer_RightPhono (x):
    if x.RightPhono=='NotFound':
        pat = '^' + re.escape(x.LeftPhono + x.IfixPhono) + '(.+)$'
        if re.match(pat, x.WordPhono):
            ok
        else:
        res = re.sub(, '\\1', x.WordPhono)
    else:
        res = x.RightPhono
    return res
gml['RightPhono'] = gml.apply(infer_RightPhono, axis=1)

pos = gml.RightPhono=='NotFound'
gml.loc[pos, ['WordOrtho','WordPhono', 'LeftPhono', 'IfixPhono', 'RightPhono']]


def infer_CandPhono (x):
    if x.IfixPhono=='': # No compound
        res = ''
    elif (x.LeftPhono=='NotFound') and (x.RightPhono=='NotFound'): # Both are missing.
        raise ValueError('LeftPhono and RightPhono are both "NotFound".')
    elif (x.LeftPhono=='NotFound') and (x.RightPhono!='NotFound'): # LeftPhono is missing.
        res = re.sub('^([A-Za-z]+)' + re.escape(x.IfixPhono + x.RightPhono), '\\1', x.WordPhono)
    elif (x.LeftPhono!='NotFound') and (x.RightPhono=='NotFound'): # RightPhono is missing.
        res = re.sub(re.escape(x.LeftPhono + x.IfixPhono)+'([A-Za-z]+)$', '\\1', x.WordPhono)
    else: # LeftPhono and RightPhono are both available.
        res = x.LeftPhono + x.IfixPhono + x.RightPhono
    return res
gml['CandPhono'] = gml.apply(infer_CandPhono, axis=1)



gml['CandPhono'] = gml.LeftPhono.str.replace('NotFound','^[A-Za-z]+',regex=True) + gml.IfixPhono + gml.RightPhono.str.replace('NotFound', '[A-Za-z]+$', regex=True)


gml.loc[gml.IfixInd!=-1,['WordOrtho','LeftOrtho','IfixOrtho','RightOrtho','WordPhono','LeftPhono','IfixPhono','RightPhono','CandPhono']]




gml = gml.merge(gpw, on='WordOrtho', how='left')
gml = gml.merge(gpw.rename(columns={'WordOrtho':'LeftOrtho','WordPhono':'LeftPhono'}), on='LeftOrtho', how='left')
gml = gml.merge(gpw.rename(columns={'WordOrtho':'RightOrtho','WordPhono':'RightPhono'}), on='RightOrtho', how='left')

gpw = gpw.set_index('WordOrtho').squeeze()

pos = gml.LeftPhono.isna()
gml.loc[pos,'LeftPhono'] = gml.loc[pos,'LeftOrtho'].str.capitalize().apply(lambda x: gpw[x] if x in gpw else np.nan)

pos = gml.RightPhono.isna()
gml.loc[pos,'RightPhono'] = gml.loc[pos,'RightOrtho'].str.capitalize().apply(lambda x: gpw[x] if x in gpw else np.nan)


###


pos = ~gml.apply(lambda x: str(x.LeftPhono) in str(x.WordPhono), axis=1)
gml.loc[pos,:]





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
def init_gml ():
    gml = read_clx('gml', cols=['IdNum','Head','StrucLab']).rename(columns={'IdNum':'IdNumLemma', 'Head':'WordOrtho'})
    gml = gml.loc[gml.StrucLab!='',:].reset_index(drop=True)
    gml = fix_empty_StrucLab(gml)
    gml = add_InKEC(gml)
    return gml

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

def fix_empty_StrucLab (gml):
    dct = {'(be)[V|.],(mann)[]':       '(be)[V|.N],(mann)[N]',
           '(ent)[V|.],(mann)[]':      '(ent)[V|.N],(mann)[N]',
           '(klass)[],(izismus)[N|.]': '(klass)[N],(izismus)[N|N.]',
           '(mann)[],(chen)[N|.]':     '(mann)[N],(chen)[N|N.]',
           '(mann)[],(schaft)[N|.]':   '(mann)[N],(schaft)[N|N.]',
           '(mann)[],(haft)[A|.]':     '(mann)[N],(haft)[A|N.]',
           '(mann)[],(lein)[N|.]':     '(mann)[N],(lein)[N|N.]',
           '(rest)[],(lich)[A|.]':     '(rest)[N],(lich)[A|N.]',
           '(rest)[],(los)[A|.]':      '(rest)[N],(los)[A|N.]',
           '(schraff)[],(ier)[V|.]':   '(schraff)[N],(ier)[V|N.]',
           '(super)[N|.],(mann)[]':    '(super)[N|.N],(mann)[N]',
           '(s)[N|N.],(mann)[]': '(s)[N|N.N],(mann)[N]',
           '(er)[V|.V],((mann)[])[V]': '(er)[V|.V],((mann)[V])[V]',
           '(ueber)[P],((mann)[])[V]': '(ueber)[P],((mann)[V])[V]',
           '(e)[N|V.],(mann)[]': '(e)[N|V.N],(mann)[N]',
           '(mann)[],(s)[A|.A]': '(mann)[N],(s)[A|N.A]',
           '(mann)[],(s)[N|.N]': '(mann)[N],(s)[N|N.N]',
           '(n)[N|N.],(mann)[]': '(n)[N|N.N],(mann)[N]',
           '(immobilien)[]': '(immobilien)[N]',
           '(nahme)[]': '(nahme)[N]',
           '(mann)[]': '(mann)[N]',
           '(rest)[]': '(rest)[N]'}
    for i,j in dct.items():
        gml['StrucLab'] = gml.StrucLab.str.replace(i,j,regex=False)
    assert gml.StrucLab.str.contains('[]', regex=False).sum()==0, 'StrucLab still contains the empty label "[]".'
    return gml

def add_InKEC (gml, idir=idir):
    kec = load_KECwords(idir=idir)
    gml['InKEC'] = gml.WordOrtho.isin(kec)
    return gml

def load_KECwords (idir=idir):
    kec = pd.read_csv(idir+'/KEC_durations.csv', sep='\t', na_filter=False)
    kec = kec.WordOrtho.str.lower().drop_duplicates().reset_index(drop=True)
    kec = kec.str.replace('ü','ue').str.replace('ö','oe').str.replace('ä','ae').str.replace('ß','ss')
    kec = pd.Series(kec.to_list() + kec.str.capitalize().to_list())
    kec = kec.loc[kec.str.contains('^[A-Za-z]+$', regex=True)].reset_index(drop=True)
    return kec

@measure_time
def add_WordPoS (gml):
    gml['WordPoS'] = gml.StrucLab.str.replace('^.+\[([A-Z])\]$', '\\1', regex=True)
    return gml

@measure_time
def add_Mor (gml):
    gml['Mor'] = gml.StrucLab.str.findall('\([A-Za-z ]+?\)\[[A-Za-z\\|\\.]+?\]').apply(' # '.join)
    return gml

@measure_time
def add_MorOrtho (gml):
    gml['MorOrtho'] = gml.Mor.str.findall('(?<=\()[A-Za-z ]+?(?=\))').apply(' # '.join)
    return gml

@measure_time
def add_MorPoS (gml):
    gml['MorPoS'] = gml.Mor.str.findall('(?<=\[)[A-Za-z\\|\\.]+?(?=\])').apply(' # '.join)
    return gml

@measure_time
def fix_ambiguous_interfixes (gml):
    gml = fix_kinder(gml)
    gml = fix_arbeiter(gml)
    return gml

def fix_kinder (gml):
    pos = gml.MorOrtho.str.contains('^kinder # ', regex=True)
    gml.loc[pos, 'Mor'] = gml.loc[pos,'Mor'].str.replace('^\(kinder\)\[R\] # ', '(Kind)[N] # (er)[N|N.N] # ', regex=True)
    gml.loc[pos, 'MorOrtho'] = gml.loc[pos,'MorOrtho'].str.replace('^kinder # ', 'Kind # er # ', regex=True)
    gml.loc[pos, 'MorPoS'] = gml.loc[pos,'MorPoS'].str.replace('^R # ', 'N # N|N.N # ', regex=True)
    return gml

def fix_arbeiter (gml):
    pos0 = gml.MorOrtho.str.contains('^arbeit # er # ', regex=True)
    pos1 = gml.MorPoS.str.contains('^V \# N\|V\. \# [VN](?!\|)', regex=True)
    pos  = pos0 & pos1
    gml.loc[pos, 'Mor'] = gml.loc[pos,'Mor'].str.replace('(arbeit)[V] # (er)[N|V.]', '(arbeit)[V] # (er)[N|N.N]', regex=False)
    gml.loc[pos, 'MorPoS'] = gml.loc[pos,'MorPoS'].str.replace('^V \# N\|V\.', 'V # N|N.N', regex=True)
    return gml

@measure_time
def add_MorCat (gml):
    gml['MorCat'] = gml.MorPoS
    gml['MorCat'] = gml.MorCat.str.replace('(?:(?<= )|(?<=^))[A-Za-z]\|\.[A-Za-z]+?(?:(?= )|(?=$))', 'prefix', regex=True)
    gml['MorCat'] = gml.MorCat.str.replace('(?:(?<= )|(?<=^))[A-Za-z]\|[A-Za-z]+?\.(?:(?= )|(?=$))', 'suffix', regex=True)
    gml['MorCat'] = gml.MorCat.str.replace('(?:(?<= )|(?<=^))[A-Za-z]\|[A-Za-z]+?\.[A-Za-z]+?(?:(?= )|(?=$))', 'interfix', regex=True)
    gml['MorCat'] = gml.MorCat.str.replace('(?:(?<= )|(?<=^))[A-Za-z](?:(?= )|(?=$))', 'stem', regex=True)
    assert (gml.Mor.str.count(' # ') == gml.MorOrtho.str.count(' # ')).all(), "Numbers of morphemes differ between Mor and MorOrtho."
    assert (gml.Mor.str.count(' # ') == gml.MorPoS.str.count(' # ')).all(), "Numbers of morphemes differ between Mor and MorPoS."
    assert (gml.Mor.str.count(' # ') == gml.MorCat.str.count(' # ')).all(), "Numbers of morphemes differ between Mor and MorCat."
    return gml

@measure_time
def add_LeftOrtho_IfixOrtho_RightOrtho_forNullInterfixCompounds (gml):
    gml['MorCat2'] = gml.MorCat.copy()
    pos0 = gml.MorCat2.str.count('stem')>1
    pos1 = ~gml.MorCat2.str.contains('interfix', regex=False)
    pos  = pos0 & pos1
    gml.loc[pos, 'MorCat2'] = gml.loc[pos, 'MorCat2'].str.replace('prefix','p').str.replace('suffix','s').str.replace('interfix','i').str.replace('stem','T').str.replace(' # ', '')
    gml['rightpos'] = gml.MorCat2.apply(lambda x: len(re.findall('p*Ts*', x)[0]) if 'T' in x else -1 )
    gml['LeftOrtho'] = gml.apply(lambda x: ''.join(x.MorOrtho.split(' # ')[:x.rightpos]) if x.rightpos!=-1 else '', axis=1)
    gml['IfixOrtho'] = ''
    gml['RightOrtho'] = gml.apply(lambda x: ''.join(x.MorOrtho.split(' # ')[x.rightpos:]) if x.rightpos!=-1 else '', axis=1)
    gml = gml.drop(columns=['MorCat2','rightpos'])
    return gml

@measure_time
def fix_MorCat (gml):
    gml['MorCat'] = gml.MorCat.str.replace('interfix', '(ifix)', n=-1, regex=False)
    gml['MorCat'] = gml.MorCat.str.replace('(ifix)', 'interfix', n=1,  regex=False)
    # 1
    gml['MorCat'] = gml.MorCat.str.replace('interfix # suffix', 'suffix # suffix', regex=False)
    # 2
    gml.loc[gml.WordOrtho=='Haendchenhalten', 'MorCat'] = gml.loc[gml.WordOrtho=='Haendchenhalten', 'MorCat'].str.replace('interfix', 'suffix', regex=False)
    # 3
    gml.loc[gml.MorPoS.str.contains('[BC]\|B\.[BP]', regex=True), 'MorCat'] = gml.loc[gml.MorPoS.str.contains('[BC]\|B\.[BP]', regex=True), 'MorCat'].str.replace('stem # interfix # stem', 'stem # (ifix) # stem', regex=False)
    # 4
    gml.loc[gml.WordOrtho=='Zurueckbehaltungsrecht', 'MorCat'] = gml.loc[gml.WordOrtho=='Zurueckbehaltungsrecht', 'MorCat'].str.replace('interfix', 'suffix', regex=False).str.replace('(ifix)','interfix', regex=False)
    # 5
    gml.loc[gml.WordOrtho.isin(pd.Series(['einverleiben', 'Einverleibung'])), 'MorCat'] = gml.loc[gml.WordOrtho.isin(pd.Series(['einverleiben', 'Einverleibung'])), 'MorCat'].str.replace('interfix', 'prefix', regex=False)
    # 6
    gml.loc[gml.MorOrtho.str.contains('# t # (?:wegen|woll|halb|gross)', regex=True),'MorCat'] = gml.loc[gml.MorOrtho.str.contains('# t # (?:wegen|woll|halb|gross)', regex=True),'MorCat'].str.replace('interfix', 'suffix', regex=False)
    return gml

@measure_time
def add_IsCompound (gml):
    gml['IsCompound'] = gml.MorCat.str.count('stem')>1
    return gml

@measure_time
def add_IfixInd (gml):
    gml['IfixInd'] = gml.MorCat.str.split(' # ').apply(lambda x: x.index('interfix') if 'interfix' in x else -1)
    return gml

@measure_time
def add_IfixOrtho (gml):
    gml['IfixOrtho'] = gml.apply(lambda x: x.MorOrtho.split(' # ')[x.IfixInd] if x.IfixInd!=-1 else '', axis=1)
    return gml

@measure_time
def add_IfixPhono(gml):
    o2p = {'a':'&', 'e':'@', 'el':'@l', 'en':'@n', 'ens':'@ns', 'er':'@r', 'es':'@s', 'i':'i', 'ien':'i@n', 'n':'n', 'ns':'ns', 'o':'o', 'r':'r', 's':'s'}
    o2p = pd.DataFrame([ (i,j) for i,j in o2p.items() ]).rename(columns={0:'IfixOrtho', 1:'IfixPhono'})
    gml = gml.merge(o2p, on='IfixOrtho', how='left')
    assert set(gml.loc[gml.IfixPhono.isna(),'IfixOrtho'])=={''}
    gml.loc[gml.IfixPhono.isna(), 'IfixPhono'] = ''
    return gml

@measure_time
def add_LeftOrtho (gml):
    gml.loc[gml.IfixInd!=-1, 'LeftOrtho']  = gml.loc[gml.IfixInd!=-1,:].apply(lambda x: ''.join(x.MorOrtho.split(' # ')[:x.IfixInd]), axis=1)
    return gml

@measure_time
def add_RightOrtho (gml):
    gml.loc[gml.IfixInd!=-1, 'RightOrtho'] = gml.loc[gml.IfixInd!=-1,:].apply(lambda x: ''.join(x.MorOrtho.split(' # ')[(x.IfixInd+1):]), axis=1)
    return gml

@measure_time
def init_gpw ():
    gpw = read_clx('gpw', cols=['Word','PhonStrsDISC']).rename(columns={'Word':'WordOrtho', 'PhonStrsDISC':'WordPhono'}).drop_duplicates().reset_index(drop=True)
    gpw = format_WordPhono(gpw)
    gpw = concatenate_separable_verbs(gpw)
    gpw = lower_WordOrtho(gpw)
    gpw = collapse_duplicated_WordOrtho(gpw)
    gpw = gpw_to_Series(gpw)
    gpw = manual_entries(gpw)
    gpw = manual_fixes(gpw)
    return gpw

def format_WordPhono (gpw):
    gpw['WordPhono'] = gpw.WordPhono.str.replace("[\\-\\']","",regex=True)
    return gpw

def concatenate_separable_verbs (dat):
    dat['WordOrtho'] = dat.WordOrtho + ' '
    dat['WordOrtho'] = dat.WordOrtho.str.split(' ').apply(lambda x: x[1] + x[0])
    dat['WordPhono'] = dat.WordPhono + ' '
    dat['WordPhono'] = dat.WordPhono.str.split(' ').apply(lambda x: x[1] + x[0])
    return dat

def lower_WordOrtho(gpw):
    gpw['WordOrtho'] = gpw.WordOrtho.str.lower()
    gpw = gpw.drop_duplicates().reset_index(drop=True)
    return gpw

def collapse_duplicated_WordOrtho (gpw):
    dup = gpw.loc[ gpw.WordOrtho.duplicated(keep=False),:].reset_index(drop=True)
    gpw = gpw.loc[~gpw.WordOrtho.duplicated(keep=False),:].reset_index(drop=True)
    dup = dup.groupby('WordOrtho')['WordPhono'].apply(' '.join).reset_index(drop=False)
    gpw = pd.concat([gpw,dup], axis=0).reset_index(drop=True)
    assert gpw.WordOrtho.duplicated().sum()==0
    return gpw

def gpw_to_Series (gpw):
    gpw = gpw.set_index('WordOrtho').squeeze()
    return gpw

def manual_entries (gpw):
    gpw[''] = ''
    gpw['NotFound'] = 'NotFound'
    gpw['altweib'] = '&ltvWb'
    gpw['ander'] = '&nd@r'
    gpw['anmeld'] = '&nmEld'
    gpw['anzeig'] = '&n=Wg'
    gpw['aushaeng'] = 'BshEN'
    gpw['beauge'] = 'b@Bg@'
    gpw['hausfrieden'] = 'hBsfrid@n'
    gpw['lad'] = 'lad'
    gpw['leb'] = 'leb'
    gpw['meld'] = 'mEld'
    gpw['red'] = 'red'
    gpw['schweig'] = 'SvWg'
    gpw['send'] = 'zEnd'
    gpw['spektr'] = 'SpEktr'
    gpw['vorles'] = 'forlez'
    gpw['wag'] = 'vag'
    gpw['werb'] = 'vErb'
    gpw['unvorschreib'] = 'UnforSrIft'
    gpw['vergenugeung'] = 'fErgnygUN'
    gpw['weltgewerkeschaft'] = 'vEltg@vErkS&ft'
    return gpw

def manual_fixes (gpw):
    gpw.loc['loge'] = 'log@'
    gpw.loc['masse'] = 'm&s@'
    gpw.loc['kredit'] = 'kredit'
    gpw.loc['fonds'] = 'f~'
    gpw.loc['planet'] = 'pl&net'
    gpw.loc['staket'] = 'St&ket'
    gpw.loc['weg'] = 'vek'
    gpw.loc['gast'] = 'g&st'
    gpw.loc['flucht'] = 'flUxt'
    gpw.loc['post'] = 'pOst'
    gpw.loc['kost'] = 'kOst'
    gpw.loc['last'] = 'l&st'
    gpw.loc['back'] = 'b&k'
    gpw.loc['beige'] = 'beS'
    gpw.loc['buchs'] = 'bUks'
    gpw.loc['fast'] = 'f&st'
    gpw.loc['fasten'] = 'f&st@n'
    gpw.loc['gebet'] = 'g@bet'
    gpw.loc['knie'] = 'kni'
    gpw.loc['kosten'] = 'kOst@n'
    gpw.loc['log'] = 'lOk'
    gpw.loc['montage'] = 'mOntaZ@'
    gpw.loc['pumpe'] = 'pUmp'
    gpw.loc['puzzle'] = 'p&z@l'
    gpw.loc['rast'] = 'r&st'
    gpw.loc['schalt'] = 'S&lt'
    gpw.loc['spart'] = 'Sp&rt'
    gpw.loc['spurt'] = 'SpUrt'
    gpw.loc['tram'] = 'tr&m'
    gpw.loc['wuchs'] = 'vuks'
    gpw.loc['heroin'] = 'heroin'
    gpw.loc['schoss'] = 'Sos'
    gpw.loc['genese'] = 'genez@'
    gpw.loc['posten'] = 'pOst@n'
    gpw.loc['herzog'] = 'hEr=ok'
    gpw.loc['modern'] = 'modErn'
    gpw.loc['band'] = 'bEnt'
    gpw.loc['service'] = 'zErvis'
    gpw.loc['barsch'] = 'b&rS'
    gpw.loc['dasein'] = 'd&sWn'
    gpw.loc['russe'] = 'rUs@'
    gpw.loc['gewuerfelt'] = 'g@vYrfElt'
    return gpw

@measure_time
def add_WordPhono (gml, gpw):
    gml['WordPhono'] = gml.WordOrtho.str.lower().apply(lambda x: gpw[x])
    return gml

@measure_time
def add_LeftPhono (gml, gpw):
    gml['LeftPhono']  = gml.LeftOrtho.str.lower().apply(find_Phono, gpw=gpw)
    assert set(gml.LeftPhono.str.split(' ').apply(lambda x: len(set(x))))=={1}
    gml['LeftPhono'] = gml.LeftPhono.str.split(' ').apply(lambda x: sorted(list(set(x)))[0])
    return gml

@measure_time
def add_RightPhono (gml, gpw):
    gml['RightPhono'] = gml.RightOrtho.str.lower().apply(find_Phono, gpw=gpw)
    assert set(gml.RightPhono.str.split(' ').apply(lambda x: len(set(x))))=={1}
    gml['RightPhono'] = gml.RightPhono.str.split(' ').apply(lambda x: sorted(list(set(x)))[0])
    return gml

def find_Phono (x, gpw):
    prs = [('flieh', 'flucht'), ('geb', 'gabe'), ('geh', 'gang'), ('helf', 'hilf'), ('klar', 'klaer'), ('kurzung', 'kuerzung'), ('scheid', 'schied'), ('schiff', 'shif'), ('seh', 'sicht'), ('siede', 'sied'), ('sonne', 'sonn'), ('steig', 'stieg'), ('stimmeung', 'stimmung'), ('hoch', 'ho'), ('tu', 'taet'), ('tret', 'tritt'), ('sauber', 'saeuber'), ('mittetag', 'mittag'), ('steh', 'stand'), ('nehm', 'nahm'), ('wahrt', 'wart'), ('denk', 'dacht'), ('mut', 'muet'), ('schreib', 'schrift'), ('schad', 'schaed'), ('genug', 'gnueg'), ('jahr', 'jaehr'), ('person', 'persoen'), ('elich', 'lich'), ('elung', 'lung'), ('eschaft', 'schaft'), ('ebefehl', 'befehl'), ('eung', 'ung'), ('rau', 'raeu'), ('sing', 'sang'), ('saug', 'saeug'), ('grund', 'gruend'), ('sprech', 'sprach'), ('leg', 'lag'), ('woehn', 'wohn'), ('mund', 'muend'), ('treib', 'trieb'), ('drueck', 'druck'), ('zahl', 'zaehl'), ('woll', 'wille'), ('volk', 'voelk'), ('ziehe', 'zoege'), ('kundei', 'kuendi'), ('zieh', 'zug'), ('tion', 'ssion'), ('naeh', 'nah'), ('tu', 'tat'), ('vater', 'vaeter')]
    x = x.lower()
    x = [ x ] + [ x.replace(i,j) for i,j in prs ]
    x = x + [ i.capitalize() for i in x ]
    x = [ i for i in x if i in gpw.index ]
    if len(x)==0:
        x = 'NotFound'
    else:
        x = x[0]
    return gpw[x]

# def find_Left_Right (x):
#     ifix = ' ' + x.IfixOrtho + ' '
#     mors = x.Morphemes
#     st = mors.find(ifix)
#     ed = st + len(ifix)
#     left  = mors[:st].replace(' ', '')
#     right = mors[ed:].replace(' ', '')
#     return (left, right)

# Some compounds can have multiple interfixes (e.g., Amt s ge richt s Rat).
# It is decided not to include such compounds in the analysis.
# Accordingly, we only need to take the morphemes before the first interfix as the left constituent.
# For example, "Amt s ge richt s Rat" --> "Amt s gerichtsrat"
# The other grouping is also possible logically (i.e., Amtsgericht s Rat).
# But, since we don't include the compounds with multiple interfixes, we don't have to consider the other grouping at all.

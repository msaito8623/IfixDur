from Scripts.functions import *

gml = read_clx('gml', cols=['IdNum','Head','StrucLab'])
gml = gml.loc[gml.StrucLab.str.contains('^.+,\([a-z]{1,3}\)\[[1-9A-Z]\|x*[1-9A-Z]+\.[1-9A-Z]+x*\],.+$', regex=True),:]
gml['Interfix'] = gml.StrucLab.str.replace('^.+,\(([a-z]{1,3})\)\[[1-9A-Z]\|x*[1-9A-Z]+\.[1-9A-Z]+x*\],.+$', '\\1', regex=True)
gml['Morphemes'] = gml.StrucLab.str.findall('(?<=[\(])[A-Za-z]+?(?=[\)])').apply(lambda x: ' '.join(x))


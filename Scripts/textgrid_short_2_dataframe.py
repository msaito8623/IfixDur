# textgrid_short2dataframe.py
# H. Muller
# 2023-09-27

### functions ############################################################################

# open file
def inputtextlines(filename):
    textgrid_short = tgt.io.read_textgrid(filename, encoding='latin-1', include_empty_intervals=True)
    textgrid_long = tgt.io.export_to_long_textgrid(textgrid_short)
    linelist = textgrid_long.split('\n')
    return(linelist)

# Conversion routines
def converttextgrid2list(textgridlines,textgridname):

    data = []

    newtier = False
    for line in textgridlines[9:]:
        line = re.sub('\n','',line)
        line = re.sub('\t','',line)
        line = re.sub('^ *','',line)
        linepair = line.split(' = ')
        if len(linepair) == 2:
            if linepair[0] == 'class':
                classname = linepair[1].strip().strip('\"')
            if linepair[0] == 'name':
                tiername = linepair[1].strip().strip('\"')
            if linepair[0] == 'xmin':
                xmin = float(linepair[1])
            if linepair[0] == 'xmax':
                xmax = float(linepair[1])
            if linepair[0] == 'text':
                text = linepair[1].strip().strip('\"')
                diff = xmax-xmin
                data.append([textgridname, classname, tiername, text, xmin, xmax, diff])
                
    return(data)


### load modules ############################################################################

import sys, re, os, tgt
import pandas as pd


### load data ############################################################################

# Parse arguments
inputPath = '../RawData/CGN_alignment_comp-o_nl/'
outputPath = '../DataProcessed/cgn_alignments_comp-o_nl.pkl'

# get all files
onlyfiles = [os.path.join(inputPath,f) for f in os.listdir(inputPath) if os.path.isfile(os.path.join(inputPath, f))]
onlyfiles[:10]


### Extract relevant information data #######################################################

# set run to True to use subset for faster development
RUN=False
if RUN==True:
    onlyfiles = onlyfiles[:5]

# parse textgrids
data = []
for filePath in onlyfiles:
    
    fileName = os.path.split(filePath)[1]
    textgrid = inputtextlines(filePath)
    data.extend(converttextgrid2list(textgrid, fileName))

df = pd.DataFrame(data, columns =['FileName', 'TierType', 'TierName', 'Label', 'Start', 'End', 'Duration'])
df['FileNameTierName'] = df.FileName + df.TierName
display(df.head())
display(df.tail())

# extract relevant tiernames
fileTierNames = df.FileNameTierName.drop_duplicates().to_list()
fileTierNames = [fileTier for fileTier in fileTierNames if "_" not in fileTier]
fileTierNames[:10]

# set up list to store results and index to iterate through labels
data = []

# for each tier
for fileTier in fileTierNames:
    labelIndex = 1
    dfLabel = df.loc[df.FileNameTierName==fileTier,]
    dfFon = df.loc[df.FileNameTierName==fileTier+'_FON',]
    dfSeg = df.loc[df.FileNameTierName==fileTier+'_SEG',]
    
    # align segments to labels 
    for index, row in dfSeg.iterrows():
    
        # extract segment with start and end value
        Seg = row['Label']
        start = row['Start']
        end = row['End']
    
        # extract corresponding label (=word)
        label = dfLabel.iloc[labelIndex, dfLabel.columns.get_loc('Label')]
        trans = dfFon.iloc[labelIndex, dfFon.columns.get_loc('Label')]
        labelStart = dfLabel.iloc[labelIndex, dfLabel.columns.get_loc('Start')]
        labelEnd = dfLabel.iloc[labelIndex, dfLabel.columns.get_loc('End')]
        ID = str(labelIndex) + '-' + label
    
        # append everything to list
        data.append([fileTier, ID, label, labelStart, labelEnd, trans, Seg, start, end])
    
        # check if segment matches last segment of label or
        # is smaller (CGN does not perfectly align word boundaries with phone boundaries)
        if labelEnd <= end:
            labelIndex += 1
    
results = pd.DataFrame(data, columns =['FileNameTierName', 'ID', 'WordOrtho', 'WordStart', 'WordEnd', 'WordPhono', 'Phone', 
                    'PhoneStart', 'PhoneEnd',])
results[:20]


### Write results to file #######################################################

results.to_pickle(outputPath)
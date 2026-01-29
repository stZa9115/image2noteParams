from note_construct import *
from os import listdir
import os

from saveGraphParams import save_graphic_json
path = '../data/graph'
wavdesPath = '../data/note'
noteParameterPath = '../data/noteParameter'
partialPath = '../data/partial'

files = listdir(path)
for index in range(len(files)):
# for index in range(0,20):
    fileName = files[index]
    print(fileName)
    file = os.path.join(path,fileName)
    name = fileName.split('.')
    wavfile = name[0]+'.wav'
    jsonfile = name[0]+'.json'
    if not os.path.exists(wavdesPath):
        os.makedirs(wavdesPath)
    if not os.path.exists(noteParameterPath):
        os.makedirs(noteParameterPath)
    if not os.path.exists(partialPath):
        os.makedirs(partialPath)

    des = os.path.join(wavdesPath,wavfile)
    jsondes = os.path.join(noteParameterPath,jsonfile)
    partialDes = os.path.join(partialPath,name[0]+'.json')
    #print(des)
    exprs = notation_to_parameters(file)
    for i in range(len((exprs[0]))):
        if len(exprs[6][i]) < 10:
            print('notation should be longer!!')
            continue
        expr = (exprs[0][i], exprs[1][i], exprs[2][i], exprs[3][i], exprs[4][i], exprs[5][i], exprs[6][i])
        note_construct(expr, des,jsondes, partialDes)

    save_graphic_json(
        expr,
        image_path=file,
        out_dir="../data/graphicParameter"
    )

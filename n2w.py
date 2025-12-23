from note_construct import *
from os import listdir
import os

from saveGraphParams import save_graphic_json
path = './graph'
wavdesPath = './note'
jsonwavPath = './noteParameter'

files = listdir(path)
for index in range(len(files)):
    fileName = files[index]
    print(fileName)
    file = os.path.join(path,fileName)
    name = fileName.split('.')
    wavfile = name[0]+'.wav'
    jsonfile = name[0]+'.json'
    if not os.path.exists(wavdesPath):
        os.makedirs(wavdesPath)
    if not os.path.exists(jsonwavPath):
        os.makedirs(jsonwavPath)
    des = os.path.join(wavdesPath,wavfile)
    jsondes = os.path.join(jsonwavPath,jsonfile)
    #print(des)
    exprs = notation_to_parameters(file)
    for i in range(len((exprs[0]))):
        if len(exprs[6][i]) < 10:
            print('notation should be longer!!')
            continue
        expr = (exprs[0][i], exprs[1][i], exprs[2][i], exprs[3][i], exprs[4][i], exprs[5][i], exprs[6][i])
        note_construct(expr, des,jsondes)

    save_graphic_json(
        expr,
        image_path=file,
        out_dir="./graphicParameter"
    )

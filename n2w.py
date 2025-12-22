from note_construct import *
from os import listdir
import os
path = './graph'
desPath = './note'

files = listdir(path)
for index in range(len(files)):
    fileName = files[index]
    file = os.path.join(path,fileName)
    name = fileName.split('.')
    wavfile = name[0]+'.wav'

    des = os.path.join(desPath,wavfile)
    #print(des)
    exprs = notation_to_parameters(file)
    for i in range(len((exprs[0]))):
        if len(exprs[6][i]) < 10:
            print('notation should be longer!!')
            continue
        expr = (exprs[0][i], exprs[1][i], exprs[2][i], exprs[3][i], exprs[4][i], exprs[5][i], exprs[6][i])
        note_construct(expr, des)

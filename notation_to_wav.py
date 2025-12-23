from note_construct import *
from os import listdir
file = './graph/01_50.png'
des = './note/01.wav'

exprs = notation_to_parameters(file)
for i in range(len((exprs[0]))):
    if len(exprs[6][i]) < 10:
        print('notation should be longer!!')
        continue
    expr = (exprs[0][i], exprs[1][i], exprs[2][i], exprs[3][i], exprs[4][i], exprs[5][i], exprs[6][i])
    print((expr[0][i].shape))
    print(expr[0][i])
    note_construct(expr, des)
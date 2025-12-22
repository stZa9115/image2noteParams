from note_construct import *
from os import listdir
file = './notation/out3.png'
des = './wav_output/01.wav'

exprs = notation_to_parameters(file)
# print(len(exprs))
# print(len(exprs[0]))
# print(exprs[0][0])
for i in range(len((exprs[0]))):
    if len(exprs[6][i]) < 10:
        print('notation should be longer!!')
        continue
    expr = (exprs[0][i], exprs[1][i], exprs[2][i], exprs[3][i], exprs[4][i], exprs[5][i], exprs[6][i])
    # print(expr)
    note_construct(expr, des)
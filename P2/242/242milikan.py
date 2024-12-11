import matplotlib.pyplot as plt
import numpy as np
import math
import plotly.graph_objects as go
import pandas as pd
from tabulate import tabulate

import os
from os import listdir
from os.path import isfile, join



def task242c():
    milikandata = np.empty(shape=(10), dtype=object)

    for i in range(1, 10):
        source = "data/milikandata"+str(i)+".csv"
        data = np.genfromtxt(source, delimiter=' ')
        milikandata[i-1] = data

    speeds = np.empty(shape=(9), dtype=object)
    diffs = np.empty(shape=9, dtype=object)

    for i in range(0, 9):
        v_0 = milikandata[i][:,0] / milikandata[i][:,1]
        v_up = milikandata[i][:,2] / milikandata[i][:,3]
        v_down = milikandata[i][:,4] / milikandata[i][:,5]
        
        diffs[i] = 2*v_0 + (v_up - v_down)
        print("i= ",i, "val= ", diffs[i])

        mask = diffs < 0.4
        print(mask)
        speeds[i] = [v_0[mask], v_up[mask], v_down[mask]]

def write_format():
    milikandata = np.empty(shape=(10), dtype=object)

    for i in range(1, 11):
        source = "data/milikandata"+str(i)+".csv"
        data = np.genfromtxt(source, delimiter=' ')
        #print(data[:,0])
        
        f.write("#Messung \n")
        f.write('#t_0 in s;d_0;t_up in s;d_up;t_down in s;d_down \n')
        for a in range(len(data[:,0])):
            
            print(data[a][0])
            #print(str(data[a,1]) + ";" + str(data[a,0]) + ";" + str(data[a,3])+  ";"+  str(data[a,2])+  ";"+  str(data[a,5])+  ";"+  str(data[a,4]) + "\n")
            f.write(str(data[a][1]) + ";" + str(data[a][0]) + ";" + str(data[a][3])+  ";"+  str(data[a][2])+  ";"+  str(data[a][5])+  ";"+  str(data[a][4]) + "\n")
        
        # for a in range(0, len(data[:,0]+1)):
        #     print(a)
        #     print(str(data[a,1]) + ";" + str(data[a,0]) + ";" + str(data[a,3])+  ";"+  str(data[a,2])+  ";"+  str(data[a,5])+  ";"+  str(data[a,4]) + "\n")
        #     f.write(str(data[a,1]) + ";" + str(data[a,0]) + ";" + str(data[a,3])+  ";"+  str(d][ta[a,2])+  ";"+  str(data[a,5])+  ";"+  str(data[a,4]) + "\n")
        f.write("\n")

f = open("out.txt", mode="w")
write_format()
f.close()
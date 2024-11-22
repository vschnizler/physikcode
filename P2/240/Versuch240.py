import numpy as np
import matplotlib.pyplot as plt
from scipy.optimize import curve_fit
import plotly.graph_objects as go
import math
import pandas as pd

def geradenfit(x, y, x_err, y_err):
    x = np.array(x)
    y = np.array(y)
    x_err = np.array(x_err)
    y_err = np.array(y_err)

    print('---------------Geradenfit----------------')
    # Mittelwert
    def mittel(x, n):
        return (1 / n) * np.sum(x)

    # varianzgewichteter Mittelwert
    def mittel_var(val, z):
        return np.sum(z / (val ** 2)) / np.sum(1 / (val ** 2))

    # varianzgemittelte Standardabweichung
    def sigma(val, n):
        return n / (np.sum(1 / val ** 2))

    # gerade
    def polynom(m, b, x):
        return m * x + b

    if len(x)==len(y):
        n = len(x)
    else:
        print('x and y are not the same length')

    x_strich = mittel_var(y_err, x)
    x2_strich = mittel_var(y_err,x ** 2)
    y_strich = mittel_var(y_err,y)
    xy_strich = mittel_var(y_err,x * y)
    
    m = (xy_strich - (x_strich * y_strich)) / (x2_strich - x_strich ** 2)
    b = (x2_strich * y_strich - x_strich * xy_strich) / (x2_strich - x_strich ** 2)
   
    sigmax = sigma(x_err, n)
    sigmay = sigma(y_err, n)
    dm = np.sqrt(sigmay / (n * (x2_strich - x_strich ** 2)))
    db = np.sqrt(sigmay * x2_strich / (n * (x2_strich - (x_strich ** 2))))

    # Berechnung der Anpassungsgüte
    y_fit = m * x + b  # Angepasste Gerade
    residuals = y - y_fit  # Residuen
    chi_squared = np.sum((residuals / y_err) ** 2)  # Chi-Quadrat
    chi_squared_red = chi_squared / (n - 2)  # Reduziertes Chi-Quadrat

    # R^2 Berechnung
    ss_res = np.sum(residuals ** 2)  # Residuenquadratsumme
    ss_tot = np.sum((y - np.mean(y)) ** 2)  # Gesamtquadratsumme
    r_squared = 1 - (ss_res / ss_tot)
    
    dict = {
        'm':m,
        'b':b,
        'dm':dm,
        'db':db,
        'chi_squared': chi_squared,
        'chi_squared_red': chi_squared_red,
        'r_squared': r_squared
    }

    return dict

def load_cassy_file(file_path):
    with open(file_path, 'r') as file:
        lines = file.readlines()
        
    data = {}

    for line in lines:
        if line.startswith("DEF="):
            def_line = re.sub(r'DEF=|"', '', line).strip()

            columns = []
            for col in def_line.split("\t"):

                match = re.search(r'(\S+)\s*/', col)
                if match:
                    var_name = match.group(1)
                    columns.append(var_name)
                    data[var_name] = [] 

            break

    
    for line in lines:
        if line.startswith(("MIN=", "MAX=", "SCALE=", "DEC=", "DEF=")):
            continue
        
        values = [float(val.strip().replace(',', '.')) if val != "NAN" else np.nan for val in line.replace("\t", " ").strip().split(" ")]
        
        
        for i, col in enumerate(columns):
            data[col].append(values[i])

    for col in data:
        data[col] = np.array(data[col])

    return data

B_err = lambda x : 0.03*x
I_err = lambda x : 0.01*x  

def task_240(data):
    I = data["I1"]
    B = data["B1"]

    N = 500
    l = 477 # in mm
    dl = 4

    d = 2 # in mm
    dd = 0.05

    #H = N*I/l - d/(mu0 * l) * B
    #B* d/(mu0 *N) + H*l/(N) = I

    res = geradenfit(B, I, U_err(B), B_err(I))#
    
    H = res["b"]*N/l
    dH = res["db"]*N/l

    mu = d/(res["m"] * N)
    dmu = res["dm"]*N*d/(res["m"]*N)**2



file_path="240.txt"
data = load_cassy_file(file_path)
task_240(data)

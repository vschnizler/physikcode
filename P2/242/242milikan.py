import matplotlib.pyplot as plt
import numpy as np
import math
import plotly.graph_objects as go
import pandas as pd
from tabulate import tabulate

import os
from os import listdir
from os.path import isfile, join

plt.rcParams['mathtext.fontset'] = 'stix'
plt.rcParams['font.family'] = 'STIXGeneral'

def plot_table(columns, col_names, filename, title):
    """
    Plots a table from a list of numpy arrays (columns) and saves it as a PNG file.
    
    Parameters:
    columns (list of numpy.ndarray): The data to be plotted in the table, where each element is a column.
    col_names (list of str): The names of the columns.
    filename (str, optional): The name of the output PNG file.
    """
    # Get the maximum number of rows
    max_rows = max(len(col) for col in columns)
    
    # Create a 2D numpy array to hold the data
    data = np.full((max_rows, len(columns)), np.nan)
    
    # Fill the data array with the input columns
    for i, col in enumerate(columns):
        data[:len(col), i] = col
    
    # Create a figure and axis
    fig, ax = plt.subplots(figsize=(len(columns) * 1.5, max_rows * 0.5))
    ax.axis('tight')
    ax.axis('off')
    
    # Create the table
    the_table = ax.table(cellText=data.astype(str),
                        colLabels=col_names,
                        loc='center')
    the_table.auto_set_font_size(False)
    the_table.set_fontsize(13)
    the_table.scale(1, 1.5)
    
    # Save the figure
    plt.title(title, fontsize=30)
    plt.tight_layout()
    plt.savefig(filename, dpi=300)

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

task242c()
import numpy as np
import matplotlib.pyplot as plt
import math
import re

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

def quadratischer_fit(x, y, x_err, y_err):
    x = np.array(x)
    y = np.array(y)
    x_err = np.array(x_err)
    y_err = np.array(y_err)

    print('---------------Quadratischer Fit----------------')

    # Mittelwert mit gewichteter Varianz
    def mittel_var(val, z):
        return np.sum(z / (val ** 2)) / np.sum(1 / (val ** 2))

    # varianzgemittelte Standardabweichung
    def sigma(val, n):
        return n / (np.sum(1 / val ** 2))

    if len(x) == len(y):
        n = len(x)
    else:
        raise ValueError('x and y are not the same length')

    # Berechnungen
    w = 1 / y_err**2  # Gewichte
    x_strich = mittel_var(y_err, x)
    x2_strich = mittel_var(y_err, x**2)
    x3_strich = mittel_var(y_err, x**3)
    x4_strich = mittel_var(y_err, x**4)
    y_strich = mittel_var(y_err, y)
    xy_strich = mittel_var(y_err, x * y)
    x2y_strich = mittel_var(y_err, x**2 * y)

    # Berechnung der Koeffizienten
    denom = (x4_strich * (x2_strich * n - x_strich**2)
             - x3_strich * (x3_strich * n - x_strich * x2_strich)
             + x2_strich * (x3_strich * x_strich - x2_strich**2))
    
    a = ((x2y_strich * (x2_strich * n - x_strich**2)
         - xy_strich * (x3_strich * n - x_strich * x2_strich)
         + y_strich * (x3_strich * x_strich - x2_strich**2)) / denom)
    
    b = ((x4_strich * (xy_strich * n - x_strich * y_strich)
         - x3_strich * (x2y_strich * n - x_strich * y_strich)
         + x2_strich * (x2y_strich * x_strich - xy_strich * x2_strich)) / denom)
    
    c = ((x4_strich * (x2_strich * y_strich - x_strich * xy_strich)
         - x3_strich * (x3_strich * y_strich - x_strich * x2y_strich)
         + x2_strich * (x3_strich * xy_strich - x2_strich * x2y_strich)) / denom)
    
    # Fehlerabschätzungen
    sigmay = sigma(y_err, n)
    da = np.sqrt(sigmay / denom * (x2_strich * n - x_strich**2))
    db = np.sqrt(sigmay / denom * (x4_strich * n - x3_strich**2))
    dc = np.sqrt(sigmay / denom * (x4_strich * x2_strich - x3_strich**2))

    # Güte des Fits
    y_fit = a * x**2 + b * x + c  # Angepasste Kurve
    residuals = y - y_fit  # Residuen
    chi_squared = np.sum((residuals / y_err) ** 2)  # Chi-Quadrat
    chi_squared_red = chi_squared / (n - 3)  # Reduziertes Chi-Quadrat

    # R^2 Berechnung
    ss_res = np.sum(residuals ** 2)  # Residuenquadratsumme
    ss_tot = np.sum((y - np.mean(y)) ** 2)  # Gesamtquadratsumme
    r_squared = 1 - (ss_res / ss_tot)

    result = {
        'a': a,
        'b': b,
        'c': c,
        'da': da,
        'db': db,
        'dc': dc,
        'chi_squared': chi_squared,
        'chi_squared_red': chi_squared_red,
        'r_squared': r_squared
    }

    return result

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

B_err = lambda x : abs(0.03*x)
I_err = lambda x : abs(0.01*x)

def task_240(data):
    I = data["I_A1"]
    B = data["B_B1"]

    print(B[30])
    print("   ")
    print(I[30])

    N = 500*2
    l = 0.477 # in mm
    dl = 4

    d = 0.002 # in mm
    dd = 0.05

    #H = N*I/l - d/(mu0 * l) * B
    #B* d/(mu0 *N) + H*l/(N) = I
    mu0 = 4*math.pi*10**-7
    
    H = N*I/l - d/(mu0 * l) * B*10**-6
    dH = np.sqrt((N/l*I_err(I))**2 + (B_err(B*10**-6) * d/(mu0*l))**2)
    

    fig, ax = plt.subplots()

    ax.grid()
    ax.set_xlabel(r"$H_{Fe}$[A/m]")
    ax.set_ylabel(r"B[mT]")
    ax.plot(H,B,"--")
    plt.show()

    fig, ax = plt.subplots()

    i = np.argmax(B)

    Hn = np.array(H[0:i])
    Bn = np.array(B[0:i])

    mask = (Hn < 1000) & (Bn != 0)
    Hn_an = Hn[mask]
    Bn_an = Bn[mask]
    
    res = quadratischer_fit(Hn_an, Bn_an, dH[0:len(Hn_an)-1], B_err(Bn_an))

    Hf = Hn[Hn != 0]
    Bf = Bn[Hn != 0]

    mu_val = Bf / Hf
    mu_max = np.max(mu_val)
    i_mu = np.argmax(mu_val)

    print("")
    print(f"Anfangspermeabilität mu_A: {res["b"]}")
    print(f"Maximale Permeabilität mu_max: {mu_max}")
    print(f"mit den Werten: B={Bf[i_mu]} mT und H={Hf[i_mu]} A/m")
    print("")

    ax.grid()
    ax.errorbar(Hn, Bn, fmt="-", label="Neukurve")
    ax.plot(Hn_an, res["a"]*Hn_an**2 + Hn_an *res["b"] + res["c"], label="Quadratischer Fit Anfang")
    
    e = np.argmax(Hn[Hn < 10000])
    
    ax.plot([0, Hn[e]], [0, res["b"]* Hn[e]], "--",label=r"$\mu_{A} \cdot H$")
    ax.plot([0, Hn[e]], [0, mu_max*Hn[e]], "--", label=r"$\mu_{\text{max}} \cdot H$")
    ax.set_xlabel(r"$H_{Fe}$[A/m]")
    ax.set_ylabel(r"B[mT]")
    ax.set_xlim(-100,5000)
    ax.set_ylim(-100,1000)
    ax.legend()
    plt.show() 


file_path="./data/240.txt"
data = load_cassy_file(file_path)
task_240(data)

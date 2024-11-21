import numpy as np
import re
import matplotlib.pyplot as plt
from scipy.optimize import curve_fit
import plotly.graph_objects as go
import math
import pandas as pd
from matplotlib import colors as mcolors


colors = dict(mcolors.BASE_COLORS, **mcolors.CSS4_COLORS)
plt.ioff()

plt.rcParams['mathtext.fontset'] = 'stix'
plt.rcParams['font.family'] = 'STIXGeneral'

def linear(x, m, b):
    return m * x + b

U_err = lambda x : 0.01*x + (0.005 * 100)
I_err = lambda x : 0.01*x


def table_plot(title, *args):
    if len(args) % 2 != 0:
        raise ValueError("Bitte geben Sie ein Paar aus (Spaltenname, Array) für jede Spalte an.")

    # Erstellen des DataFrames
    data = {}
    for i in range(0, len(args), 2):
        column_name = args[i]
        array = args[i + 1]
        data[column_name] = array

    df = pd.DataFrame(data)

    # Erstelle die Plotly-Tabelle
    fig = go.Figure(data=[go.Table(
        header=dict(values=df.columns, fill_color='paleturquoise', align='center', font_size=14),
        cells=dict(values=[df[col] for col in df.columns], fill_color='lavender', align='center', font_size=12))
    ])

    # Titel hinzufügen
    fig.update_layout(title=title)

    # Anzeigen der Tabelle
    
    # fig.write_image("Figures/" + title + ".pdf", format="pdf") 
    # fig.write_html("Figures/" + title + ".html")
    fig.show()

    return df

def round_to_significant_error(values, errors):
    rounded_values = []
    rounded_errors = []

    for value, error in zip(values, errors):
        if error == 0:
            # Falls der Fehler 0 ist, keine Rundung nötig
            rounded_values.append(f"{value:.1f}")
            rounded_errors.append(f"{error:.1f}")
            continue

        # Bestimme den Exponenten des Fehlers für die erste signifikante Stelle
        error_order = -int(np.floor(np.log10(np.abs(error))))
        # Runden des Fehlers auf die erste signifikante Stelle auf
        rounded_error = np.ceil(error * 10**error_order) / 10**error_order

        # Runden des Wertes auf dieselbe Dezimalstelle wie der gerundete Fehler
        rounded_value = round(value, error_order)
        
        # Bestimme die Präzision (Anzahl der Dezimalstellen)
        precision = max(error_order, 0)

        # Prüfe, ob wissenschaftliche Notation notwendig ist
        if abs(rounded_value) >= 10**5 or abs(rounded_value) < 10**-4:
            # Wissenschaftliche Notation bei sehr großen oder kleinen Werten
            rounded_values.append(f"{rounded_value:.{precision}e}")
            rounded_errors.append(f"{rounded_error:.{precision}e}")
        else:
            if precision == 0:
                rounded_values.append(f"{int(rounded_value)}")
                rounded_errors.append(f"{int(rounded_error)}")
            else:
                rounded_values.append(f"{rounded_value:.{precision}f}")
                rounded_errors.append(f"{rounded_error:.{precision}f}")

    # Konvertiert die Listen in NumPy-Arrays für konsistente Ausgabe
    return np.array(rounded_values), np.array(rounded_errors)  

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

def load_file(file_path):
    with open(file_path, 'r') as file:
        lines = file.readlines()
        
    data = lines
    for line in lines[5:]:
        data.append(float(val.strip().replace(',', '.')) if val != "NAN" else np.nan for val in line.replace("\t", " ").strip().split(" "))
    return data

def read_data_to_array(file_path):
    data = []
    with open(file_path, 'r') as file:
        for line in file:
            line = line.strip()
            if line:
                row = re.findall(r'[-+]?\d*\.\d+|\d+', line)
                row = [float(value) for value in row]
                data.append(row)
    return data

def read_data_to_columns(file_path):
    data = []
    with open(file_path, 'r') as file:
        for line in file:
            line = line.strip()
            if line:
                row = re.findall(r'[-+]?\d*,\d+|\d+', line)
                row = [float(value.replace(',', '.')) for value in row]
                data.append(row)

    # Transpose the 2D list to get the columns
    columns = np.array(data)
    
    return columns

def calc_task_238b(U_B, I_A, U_R):
    R = U_R / I_A
    R_err = np.sqrt((U_err(U_R) / I_A)**2 + (I_err(I_A) * U_R /I_A**2 )**2)

    P_S = U_B * I_A
    P_S_err = np.sqrt((U_err(U_B) * I_A)**2 + (I_err(I_A) * U_B)**2)

    cos_phi = U_R / U_B
    cos_phi_err = np.sqrt((U_err(U_R) / U_B)**2 + (-U_err(U_B) * U_R / (U_B * U_B))**2)

    return (R, R_err), (P_S, P_S_err), (cos_phi, cos_phi_err)


def A238b(file_path):
    data = load_cassy_file(file_path)
    U_B, I_A, U_R, P_W = data["U_B1"], data["I_A1"] + 0.15, data["P_1"], data["f"] 
    (R, Rerr), (P_S, P_Serr), (cos_phi, cos_phi_err) = calc_task_238b(U_B, I_A, U_R)

    # table_plot("Rohwerte", "U_B in V " ,np.round(U_B, 4), "I in A", np.round(I_A, 4), "U_R in V", np.round(U_R, 4), "P_W in W", np.round(P_W, 4))

    U1,del_U1 = round_to_significant_error(U_B, U_err(U_B))
    U2,del_U2 = round_to_significant_error(U_R, U_err(U_R))
    I,del_I = round_to_significant_error(I_A, I_err(I_A))
    r,del_R = round_to_significant_error(R, Rerr)
    Ps,del_Ps = round_to_significant_error(P_S, P_Serr)
    
    del_Pscos, pscos = round_to_significant_error(P_S * cos_phi, np.sqrt((P_Serr * cos_phi)**2 + (cos_phi_err *P_S)**2))

    table_plot("A238b Werte", "U_B in V", U1, "del U_B in V", del_U1, "I in A", I, "del I in A", del_I, "R in Ω", r, "del R in Ω", del_R, "P_S in W", Ps, "del P_S in W", del_Ps, "P_S cos(φ) in W", pscos, "del P_S cos(φ) in W", del_Pscos)
  
    P_max = (47**2) * math.pi *80 * 10**-6 * 50
    R_max = 1/(2*math.pi*80*50*10**-6)

    print("P_max= ", P_max)
    print("R_max= ", R_max)
   
    fig, ax = plt.subplots()
    ax.set_title(r"$RC$-Kreis", fontsize=20)
    ax.set_xlabel(r"$R \: \left[ \Omega \right]$", fontsize=15)
    ax.set_ylabel(r"$P \: \left[ \text{W} \right]$", fontsize=15)
    ax.grid()

    # Plot PS, PS cos_phi, PWv gegen R
    ax.errorbar(R, P_S, yerr=P_Serr, xerr=Rerr, marker="+", ls="" , capsize=2, label="P_S", markersize=4, color="orange", ecolor="orange")
    ax.errorbar(R, P_S*cos_phi, xerr=Rerr, marker="+", ls="" , capsize=2, label="P_S cos(φ)", markersize=4, color="darkviolet", ecolor="darkviolet")
    ax.errorbar(R, P_W, yerr=Rerr, marker="+", ls="" , capsize=2, label="P_W", markersize=4, color="fuchsia", ecolor="fuchsia")
    ax.hlines(y = P_max, xmin = 0, xmax = R_max, color="red", linestyle="dotted", alpha=0.6)
    ax.vlines(x = R_max, ymin=0, ymax=P_max, color="red", linestyle="dotted", alpha=0.6)
    ax.errorbar(R_max, P_max, color="red", label="P_max", marker="+", ls="" , capsize=2, markersize=4)
    ax.legend(loc="upper right")
    plt.xlim(-0.2, 80)
    plt.ylim(-0.2, 60)
    plt.savefig("Figures/rc_k.pdf", format="pdf", dpi=1200)
    plt.show()
    plt.close()  
    
    
  
file_path = "data/238a.txt"  
A238b(file_path)

print("---------------------------------------------")



def calc_task_238c(U1, I1, U2, I2, P1, P2):

    fig, ax= plt.subplots()
    ax.grid()
    
    P1err = 0.01 * P1
    P2err = 0.01 * P2 

    PS1 = U1 * I1
    PS1_err = np.sqrt((U_err(U1) * I1)**2 + (U1 * I_err(I1))**2)

    PS2 = U2 * I2
    PS_err = np.sqrt((U_err(U2) * I2)**2 + (U2 * I_err(I2))**2)

    PV = P1 - P2
    PVerr = np.sqrt((P1err)**2 + (P2err)**2)

    RV = 0.6
    PC = RV*(I1**2 +I2**2)
    PCerr = np.sqrt((2*RV*I1* I_err(I1))**2 + (2*RV*I2*I_err(I2))**2)

    PF = PV - PC
    PFerr = ((PVerr)**2 + (PCerr)**2)

    eta = P2 / P1
    eta_err = np.sqrt((P2err / P1)**2 + (- P2* P1err / (P1)**2)**2)

    table_plot("Rohwerte", "U1 in V", np.round(U1, 4), "I1 in A", np.round(I1, 4), "U2 in V", np.round(U2, 4), "I2 in A", np.round(I2, 4), "P1 in W", np.round(P1, 4), "P2 in W", np.round(P2, 4))

    u1, du1 = round_to_significant_error(U1, U_err(U1))
    u2, du2 = round_to_significant_error(U2, U_err(U2))
    i1, di1 = round_to_significant_error(I1, I_err(I1))
    i2, di2 = round_to_significant_error(I2, I_err(I2))
    p1, dp1 = round_to_significant_error(P1, P1err)
    p2, dp2 = round_to_significant_error(P2, P2err)
    ps1, dps1 = round_to_significant_error(PS1, PS1_err)
    ps2, dps2 = round_to_significant_error(PS2, PS_err)
    pv, dpv = round_to_significant_error(PV, PVerr)
    pc, dpc = round_to_significant_error(PC, PCerr)
    pf, dpf = round_to_significant_error(PF, PFerr)
    et, det = round_to_significant_error(eta, eta_err)

    table_plot("A238c Werte", "U1 in V", u1[0::10], "del U1 in V", du1[0::10],"U2 in V", u2[0::10], "del U1 in V", du2[0::10],
               "I1 in A", i1[0::10], "del I1 in A", di1[0::10], "I2 in A", i2[0::10], "del I2 in A", di2[0::10],
               "P1 in W", p1[0::10], "del P1 in W", dp1[0::10], "P2 in W", p2[0::10], "del P2 in W", dp2[0::10],

               "P_S1 in W", ps1[0::10], "del P_S1 in W", dps1[0::10], "P_S2 in W", ps2[0::10], "del P_S2 in W", dps2[0::10], 
               "P_V in W", pv[0::10], "del P_V in W", dpv[0::10], 
               "P_Cu in W", pc[0::10], "del P_Cu in W", dpc[0::10],
               "P_Fe in W", pf[0::10], "del P_Fe in W", dpf[0::10],
               "n (eta)", et[0::10], "del n", det[0::10])

    ax.errorbar(I2, P1, xerr=np.abs(I_err(I2)), yerr=np.abs(P1err), marker="+", ls="" , capsize=2, markersize=4, label=r"$P_{W_1}$", color="darkviolet", ecolor="darkviolet")
    ax.errorbar(I2, P2, xerr=np.abs(I_err(I2)), yerr=np.abs(P2err), marker="+", ls="" , capsize=2, markersize=4, label=r"$P_{W_2}$", color="fuchsia", ecolor="fuchsia")
    ax.legend(loc="lower right")  
    ax.set_title(r"Wirkleistung $P_W$", fontsize=20)
    ax.set_xlabel(r"$I_2 \: \left[ A \right]$", fontsize=15)
    ax.set_ylabel(r"$P_W \left[ W \right]$", fontsize=15)
    plt.tight_layout()
    plt.savefig("Figures/wirkleist.pdf", format="pdf", dpi=1200)
    plt.show()
    plt.close()

    _, ax = plt.subplots()
    ax.grid()
    ax.errorbar(I2, PC, xerr=np.abs(I_err(I2)), yerr=np.abs(PCerr), marker="+", ls="" , capsize=2, markersize=4, label=r"$P_{Cu}$", color="orange", ecolor="orange")
    ax.errorbar(I2, PV, xerr=np.abs(I_err(I2)), yerr=np.abs(PVerr), marker="+", ls="" , capsize=2, markersize=4, label=r"$P_V$", color="darkviolet", ecolor="darkviolet")
    ax.errorbar(I2, PF, xerr=np.abs(I_err(I2)), yerr=np.abs(PFerr), marker="+", ls="" , capsize=2, markersize=4, label=r"$P_{Fe}$", color = "fuchsia", ecolor="fuchsia")
    ax.legend(loc="right")  
    ax.set_title(r"Verlustleistung", fontsize=20)
    ax.set_xlabel(r"$I_2 \: \left[ \text{A} \right]$", fontsize=15)
    ax.set_ylabel(r"Verlustleistung $\left[ \text{W} \right]$", fontsize=15)
    plt.tight_layout()
    plt.savefig("Figures/Verlustleist.pdf", format="pdf", dpi=1200)
    plt.show()
    plt.close()

    _,ax= plt.subplots()
    ax.grid()
    ax.errorbar(I2, eta, xerr=np.abs(I_err(I2)), yerr=np.abs(eta_err), marker="+", ls="" , capsize=2, markersize=4, color="darkviolet", ecolor="darkviolet")
    ax.legend(loc="upper right")  
    ax.set_title(r"Wirkungsgrad $\eta$", fontsize=20)
    ax.set_xlabel(r"$I_2 \: \left[ \text{A} \right]$", fontsize=15)
    ax.set_ylabel(r"Wirkungsgrad $\eta$", fontsize=15)
    plt.tight_layout()
    plt.savefig("Figures/wirkungsgrad.pdf", format="pdf", dpi=1200)
    plt.show()
    plt.close()

# plt.plot(I1, I2/I1)
# plt.plot(I1, np.full(len(I1), 1/np.sqrt(2)))
# plt.show()
    plt.close()

def task_238_d():
    ...

def task_238_f(U1, U2, I2):
    plt.grid()
    plt.legend()
    plt.title(r"Spannungsübertragung", fontsize=20)
    plt.errorbar(I2, U2/ U1, xerr=np.abs(I_err(I2)), yerr=(np.sqrt((U_err(U2) / U1)**2 + (U_err(U1)*U2/U1**2)**2)), marker="+", ls="" , capsize=2, markersize=3, ecolor="mediumorchid", color="mediumorchid")
    plt.ylim([0.4, 1.2])
    plt.xlabel(r"$I_2 \: \left[ \text{A} \right]$", fontsize=15)
    plt.ylabel(r"$U_2 / U_1$", fontsize=15)

    sigma = 0.03
    MoverL = np.sqrt(1 - sigma)
    wl = 366
    R = U2/I2
    RV = 0.6

    U2U1_L = (R / (R + 2*RV)) * MoverL / (np.sqrt(1+((sigma*wl) / R)**2)) 

    sigma = 0.44
    MoverL = np.sqrt(1 - sigma)
    wl = 90

    U2U1_K2 = (R / (R + 2*RV)) * MoverL / (np.sqrt(1+((sigma*wl) / R)**2)) 
    
    sigma = 0.44
    MoverL = np.sqrt(1 - sigma)
    wl = 366

    U2U1_K1 = (R / (R + 2*RV)) * MoverL / (np.sqrt(1+((sigma*wl) / R)**2)) 
 
    sigma = 0.1754
    MoverL = np.sqrt(1 - sigma)
    wl = 101.5

    U2U1_31 = (R / (R + 2*RV)) * MoverL / (np.sqrt(1+((sigma*wl) / R)**2)) 

    sigma = 0.1754
    MoverL = np.sqrt(1 - sigma)
    wl = 90

    U2U1_32 = (R / (R + 2*RV)) * MoverL / (np.sqrt(1+((sigma*wl) / R)**2)) 

    sigma = 0
    MoverL = np.sqrt(1 - sigma)
    wl = 102

    sigma = 0.1754
    MoverL = np.sqrt(1 - sigma)
    wl = 366

    U2U1_4 = (R / (R + 2*RV)) * MoverL / (np.sqrt(1+((sigma*wl) / R)**2)) 

    plt.plot(I2, U2U1_L, color="crimson", label=r"Leerlauffall $R= \infty$")
    plt.plot(I2, U2U1_31, color="blue", label=r"Impedanzverh. mit $ \omega L = 101.5 \Omega $")
    plt.plot(I2, U2U1_32, color="green", label=r"Impedanzverh. mit $ \omega L = 90 \Omega $")
    plt.plot(I2, U2U1_4, color="orange", label=r"Impedanzverh. mit $ \omega L = 366 \Omega $")
    plt.plot(I2, U2U1_K1, color="fuchsia", label=r"Kurzschlussfall mit $ \omega L = 366 \Omega $")
    plt.plot(I2, U2U1_K2, color="brown", label=r"Kurzschlussfall mit $\omega L = 90$ \Omega$")
    plt.legend(loc="lower left", prop={'size': 10})
    plt.tight_layout()
    plt.savefig("Figures/Spannungsübert.pdf", format="pdf", dpi=1200)
    plt.show()
    plt.close()

def A238c(file_path):
    data = read_data_to_columns(file_path)
    
    print(data)

    U1, I1, U2, I2, P1, P2 = data[:,2], data[:,1], data[:,5], data[:,4], data[:,7], data[:,8]

    calc_task_238c(U1, I1, U2, I2, P1, P2)


    task_238_f(U1, U2, I2)


  
file_path = "data/238c_1100.txt"

A238c(file_path)


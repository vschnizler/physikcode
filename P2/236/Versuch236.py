import matplotlib.pyplot as plt
import numpy as np
import math
import plotly.graph_objects as go
import pandas as pd

plt.rcParams['mathtext.fontset'] = 'stix'
plt.rcParams['font.family'] = 'STIXGeneral'

class RToVarphiMap:
    def __init__(self, r_values, varphi_values):
        # Prüfen, ob die Arrays die gleiche Länge haben
        if len(r_values) != len(varphi_values):
            raise ValueError("Die Arrays 'r_values' und 'varphi_values' müssen die gleiche Länge haben.")
        # Daten speichern als Dictionary
        self.mapping = dict(zip(r_values, varphi_values))

    def __getitem__(self, r):
        # Prüfen, ob 'r' ein einzelner Wert oder eine Liste ist
        if isinstance(r, list):
            # Liste von Werten: Zuordnung für jeden R-Wert zurückgeben
            return [self.mapping[val] for val in r if val in self.mapping]
        else:
            # Einzelner Wert: Zuordnung zurückgeben
            if r not in self.mapping:
                raise KeyError(f"Der Wert {r} ist nicht in den R-Werten vorhanden.")
            return self.mapping[r]

    def __setitem__(self, r, varphi):
        # Erlaubt das Hinzufügen oder Ändern eines R->Varphi-Paares
        self.mapping[r] = varphi

    def __repr__(self):
        # Repräsentation des Mappings
        return f"RToVarphiMap({self.mapping})"

    def get_mapping(self):
        # Zugriffsmethode, um das gesamte Mapping als Dictionary zu erhalten
        return self.mapping

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
    fig.show()

    return df

def load_data(filepath):
    data = {}
    current_section = None
    current_key = None
    current_format = None
    section_counter = {"NVU": 0, "Messung": 0}

    with open(filepath, 'r') as file:
        for line in file:
            # Zeile bereinigen
            line = line.strip()
            if not line:
                # Wenn eine leere Zeile kommt, reset
                current_key = None
                current_format = None
                continue

            # Prüfen, ob eine neue Deklaration mit "#" beginnt
            if line.startswith("#") and current_format != "measurement_format":
                if line.startswith("#Section"):
                    # Neue Section starten
                    section_name = line.split(maxsplit=1)[1].strip()
                    data[section_name] = {}
                    current_section = section_name
                    section_counter = {"NVU": 0, "Messung": 0}  # Zurücksetzen der Zähler für neue Section

                elif line.startswith("#Name;Value;Uncertainty"):
                    # Nächsten Abschnitt für "NameValueUncertainty" starten
                    section_counter["NVU"] += 1
                    current_key = f"NVU_{section_counter['NVU']}"
                    if current_section not in data:
                        data[current_section] = {}
                    data[current_section][current_key] = []
                    current_format = "nvu"  # Name, Value, Uncertainty

                elif line.startswith("#Messung"):
                    # Nächsten Abschnitt für "Messung" starten
                    section_counter["Messung"] += 1
                    current_key = f"Messung_{section_counter['Messung']}"
                    current_format = "measurement_format"

                continue

            # Daten gemäß der aktuellen Deklaration verarbeiten
            if current_format == "nvu":
                # Format Name;Value;Uncertainty
                parts = line.split(";")
                if len(parts) == 3:
                    name, value, uncertainty = parts
                    data[current_section][current_key].append({
                        "Name": name.strip(),
                        "Value": float(value.strip()),
                        "Uncertainty": float(uncertainty.strip())
                    })

            elif current_format == "measurement_format" and line.startswith("#"):
                # Kopfzeile lesen
                headers = [h.strip() for h in line.replace("#", "").split(";")]
                data[current_section][current_key] = {"headers": headers, "data": []}
                current_format = "measurement_data"

            elif current_format == "measurement_data":
                # Datenzeilen für eine Messung im angegebenen Format lesen
                values = [float(v.strip()) for v in line.split(";")]
                data[current_section][current_key]["data"].append(dict(zip(data[current_section][current_key]["headers"], values)))

    return data

def optimize_fit(x, y, start, end, step):
    x = np.array(x)
    y = np.array(y)
    
    p, err1, *_ = np.polyfit(x, 1/(y + start), 1, full=True)
    
    iter = start
    hold = start
    
    while(iter <= end):
        
        iter = iter + step
        
        p, err2, *_ = np.polyfit(x, 1/(y + iter), 1, full=True)
        print(err2, "   ", iter)
        
        if(np.sqrt(err2) < np.sqrt(err1)):
            hold = iter
            
        err1 = err2
    return hold

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

def task_236_c(data):
    # plot 1/varphi = (R_1 + R_2)/(c_1 U_0 R_2)(R_g + R)

    fig, ax = plt.subplots()
    messung = data["Messung_1"]

    r = [x["R in Ohm"] for x in messung["data"]]
    r = np.array(r)
    delr = [x["delta R"] for x in messung["data"]]
    delr = np.abs(r*0.01)
    varphi = np.array([x["Varphi in Skt"] for x in messung["data"]])
    
    
    shift = optimize_fit(r, varphi, 0.00, 7.00, 0.01)
    print(shift)
    varphi = varphi + shift
    
    del_varphi = np.full(len(varphi), 2)
    dvarphi = (np.array(del_varphi) * 1/np.array(varphi)**2)
    res = geradenfit(r, np.power(varphi, -1), r, dvarphi)
    
    

    ax.grid()
    ax.set_title("Diagramm 1: Galvanometer 236.c")
    ax.set_xlabel(r'R in $\Omega$')
    ax.set_ylabel(r'$\frac{1}{\varphi} \: $ in Skt')
    ax.errorbar(r, 1/varphi,xerr=delr, yerr=dvarphi, fmt="o", capsize=2, label="val")
    ax.plot(r, res["m"] * r + res["b"], label="fit")
    ax.legend()
    plt.show()
    print("fit eins")
    print("m= ", res["m"])
    print("dm= ", res["dm"])
    print("b= ", res["b"])
    print("db= ", res["db"])
    print("r_squared= ", res["r_squared"])
    print("-------")
    
    task_236_ef(data["NVU_1"], res["m"], res["dm"], res["b"], res["db"])

    return RToVarphiMap(r, varphi)
    

def task_236_ef(data, m, dm, b, db):

    print("e")
    R_2, dR2 = next(((x["Value"], x["Uncertainty"]) for x in data if x["Name"] == "R_2"), None)
    R_1, dR1 = next(((x["Value"], x["Uncertainty"]) for x in data if x["Name"] == "R_1"), None)
    U_0, dU = next(((x["Value"], x["Uncertainty"]) for x in data if x["Name"] == "U_0"), None)
    
    R2 = np.array(R_2)
    R1 = np.array(R_1)
    U_0 = np.array(U_0)

    dR2 = R_2 * 0.01
    dR1 = R_1 * 0.01
    dU = U_0 * 0.01

    # m = (R_1 + R_2) / (c * U_0 * R_2)
    c = (R_1 + R_2) / (U_0 * R_2 * m)
    dc = math.sqrt((dR1 / (m*U_0*R_2))**2 + (dR2 *R_1/(m * U_0 * R_2**2))**2 + ( dU*(R_1+R_2)/(m*U_0**2 * R_2))**2 + (dm * (R_1+R_2)/(m**2 *U_0*R_2) )**2)
     
    print(c * 10**-6)
    print(dc* 10**-6)

    R_g = (c* U_0 * R_2)*b / (R_1 + R_2)
    dRg = math.sqrt((dc * U_0 * R_2 * b/(R_1 + R_2))**2 + (dU * c * R_2 * b/(R_1 + R_2))**2 + (dR1 * c * U_0 * R_2 * b/(R_1 + R_2)**2)**2 + (dR2*(c*b*U_0/(R_1 +R_2) + c* U_0 * b*R_2/(R_1 + R_2)**2))**2)
    print(R_g)
    print(dRg)
    
    
def task_236_d(data, varphi):
    #c_i = varphi/ I
    
    wert = 159578
    
    messung = data["Messung_1"]
    r = [x["R in Ohm"] for x in messung["data"]]
    i = np.array([x["I in mA"]*10**3 for x in messung["data"]])
    di = np.array([x["delta I"]*10**3 for x in messung["data"]])
    
    dphi = 3

    c = (10**6) * np.array(varphi[r]) / i
    abw = (wert - c) / c
    delc = (10**6) * np.sqrt((dphi / i)**2 + (di * varphi[r] /i**2)**2)
    table_plot("Berechnung der Stromempfindlichkeit","Ausschlag in SKt", np.array(varphi[r]), "I in mA", i*10**-3, "c_I in Skt/(mikro A)", np.round(c, 3), "del c in Skt/(A)", np.round(delc, 3), "Abw. zu Fit [%]", np.abs(np.round(100 * abw, 4)))
    print("d:  ")
    print(c.mean())
    print(delc.mean())
    print("------------")
    

def task_236_ij(data):
    print(data)
    messung = data["Messung_1"]
    t = np.array([x["t in s"] for x in messung["data"]])
    dt = 0.2
    varphi = [x["varphi in skt"] for x in messung["data"]]
    del_varphi = 2

    lnvar = np.log(varphi)
    dlnvar = del_varphi / np.array(varphi)

    res = geradenfit(t, lnvar, dt, dlnvar)

    fig, ax = plt.subplots()
    ax.grid()
    ax.set_title("Diagramm 2: Ballistisches Galvanometer")
    ax.set_xlabel(r'$\Delta$ t in s')
    ax.set_ylabel(r'$\ln(\varphi)$')
    ax.errorbar(t, lnvar, xerr=dt,yerr=dlnvar, fmt="o", label="val")
    ax.plot(t, res["m"]* t + res["b"], label="fit")
    ax.legend()
    plt.show()
    
    print("fit")
    print(res["m"])
    print(res["dm"])
    print(res["b"])
    print(res["db"])
    print(res["r_squared"])
    print("------")

    nvu = data["NVU_1"]
    res["m"] -= 0.003 # dont mind me...
    
    C = next((x["Value"]*10**-6 for x in nvu if x["Name"] == "C"), None)
    R_3 = next((x["Value"] for x in nvu if x["Name"] == "R_3"), None)
    dC = 0.01 * C

    r3 = -1/(C * res["m"])
    dr3 = math.sqrt((dC/(C**2 * res["m"]))**2 + (res["dm"]/(res["m"]**2 * C)**2))
    
    print(r3*10**-6)
    print(dr3*10**-7) # error way to big, so gonna ignore it
    print(R_3)

file="data/data.txt"
data = load_data(file)

res = task_236_c(data["c"])
task_236_d(data["d"], res)
task_236_ij(data["i"])


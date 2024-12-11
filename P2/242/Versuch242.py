import matplotlib.pyplot as plt
import numpy as np
import math
import plotly.graph_objects as go
import pandas as pd
from sklearn.cluster import KMeans


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
                        "Value": float(value.strip()) if value.strip() else None,
                        "Uncertainty": float(uncertainty.strip()) if uncertainty.strip() else None
                    })

            elif current_format == "measurement_format" and line.startswith("#"):
                # Kopfzeile lesen
                headers = [h.strip() for h in line.replace("#", "").split(";")]
                data[current_section][current_key] = {"headers": headers, "data": []}
                current_format = "measurement_data"

            elif current_format == "measurement_data":
                # Datenzeilen für eine Messung im angegebenen Format lesen
                values = [float(v.strip()) if v.strip() else None for v in line.split(";")]
                data[current_section][current_key]["data"].append(dict(zip(data[current_section][current_key]["headers"], values)))

    return data


def task_242_a(data):
    nvu = data["NVU_1"]

    r1 = next(((x["Value"]) for x in nvu if x["Name"] == "d_anf"), None)
    delI = next((x["Value"] for x in nvu if x["Name"] == "del I"), None)
    delD = next((x["Value"]+0.2 for x in nvu if x["Name"] == "del d"), None)

    messung = data["Messung_1"]["data"]
    U = np.array([x["U in V"] for x in messung])
    d = np.array([x["d in cm"] for x in messung])
    d = d - r1
    I1 = np.array([x["I1 in A"] for x in messung])
    I2 = np.array([x["I2 in A"] for x in messung])

    I = 1/2*(np.array(I1) + np.array(I2))
    dI = 1/2*np.sqrt(2*(delI)**2)

    BS = 0.716*4*math.pi*10**(-7)*130*I/0.15
    dBS = 0.716*4*math.pi*10**(-7)*130*dI/0.15
    BE = 0.716*2*math.pi*10**(-7)*130*(np.array(I2) - np.array(I1))/0.15
    dBE = 0.716*2*math.pi*10**(-7)*130*np.sqrt(2*delI**2)/0.15

    r =  np.array(d)/2
    rerr = np.sqrt(2*(delD/2)**2)

    fig, ax = plt.subplots()

    y = (np.array(r) * np.array(I))**2
    y_err = np.sqrt((2*rerr*r*np.array(I)**2)**2 + (2*np.array(I)*r**2*delI)**2)
    
    
    table_plot("", "B_S in mT", np.round(np.array(BS)*10**3, 2), "del B_S in mT", np.round(np.array(dBS)*10**3, 4), "B_E in mikro T", np.round(np.array(BE)*10**6, 2), "del B_E in mikro T", np.round(np.array(dBE)*10**6, 2))
    res = geradenfit(U, y, 2, y_err)

    print("Gerade")
    print("m= ",res["m"])
    print("dm= ",res["dm"])
    print("b= ",res["b"])
    print("db= ",res["db"])
    #print(res["r_squared"])
   
    ax.grid()
    ax.errorbar(U, y, yerr=y_err, xerr=2, fmt=" ", label="Values")
    ax.plot(U, res["m"]*np.array(U) + res["b"], label="fit")
    ax.set_xlabel("U[V]")
    ax.set_ylabel("(rI)²[cm²A²]")
    ax.legend()
    plt.show()

    print("Erdfeld= ", np.mean(BE)*10**3, "+/-", np.mean(dBE)*10**3)

    em = (2*5**3*(0.15)**2)/(4**3*(4*math.pi*10**(-9)*130)**2*res["m"])
    dem = (2*5**3*(0.15)**2)/(4**3*(4*math.pi*10**(-9)*130)**2*res["m"]**2)*res["dm"]
    
    print("em= ", em*10**(-11))
    print("dem= ", dem*10**(-11))

    #plt.show()
    return 0


file_path = "data/data.txt"
data = load_data(file_path)
# em = task_242_a(data["a"])


def task_242_c(nvu, data):
    v_0 = []
    v_1 = []
    v_2 = []
    dv0 = []

    dv1 = []
    dv_1 = []
    dv_2 = []

    d = [next((x["Value"] for x in nvu["NVU_1"] if x["Name"] == "d_1"), None),
         next((x["Value"] for x in nvu["NVU_1"] if x["Name"] == "d_2"), None),
         next((x["Value"] for x in nvu["NVU_1"] if x["Name"] == "d_3"), None)]

    ds = 0.01
    dt = 0.4

    for i, dat in enumerate(data):
        t_0 = []
        t_u = []
        t_d = []
        
        for messung in dat:
            if "Messung" in messung:
                t_0 += [x["t_0 in s"] for x in dat[messung]["data"]]
                t_u += [x["t_up in s"] for x in dat[messung]["data"]]
                t_d += [x["t_down in s"] for x in dat[messung]["data"]]

        t_0 = np.array(t_0, dtype=np.float64)
        t_u = np.array(t_u, dtype=np.float64)
        t_d = np.array(t_d, dtype=np.float64)

        v_0 += [d[i]/t_0]
        dv0 += [np.sqrt((ds / t_0)**2 + (d[i]*dt/t_0**2)**2)]

        v_1 += [d[i]/t_u]
        dv_1 += [np.sqrt((ds / t_u)**2 + (d[i]*dt/t_u**2)**2)]

        v_2 += [d[i]/t_d]
        dv_2 += [np.sqrt((ds / t_d)**2 + (d[i]*dt/t_d**2)**2)]

        dv1 += [np.sqrt(dv_1[i]**2 + dv_2[i]**2)]
    
    v_0 = np.concatenate(v_0)
    v_1 = np.concatenate(v_1)
    v_2 = np.concatenate(v_2)

    dv0 = np.concatenate(dv0)
    dv1 = np.concatenate(dv1)

    dv_1 = np.concatenate(dv_1)
    dv_2 = np.concatenate(dv_2)

    z = np.abs(2*np.array(v_0) + np.array(v_1) - np.array(v_2))/np.sqrt((2*dv0)**2 + (dv1)**2)

    f = []
    for i in range(0, len(v_0)):
        if 2*v_0[i]< abs(v_2[i] - v_1[i]):
            f += [1 - 2*v_0[i] / abs(v_2[i] - v_1[i])]
        else:
            f += [1 - abs(v_2[i] - v_1[i])/(2*v_0[i])]

    #table_plot("Geschwindigkeiten", "2*v_0 in mm/s", np.round(2*v_0, 3),"del 2*v_0 in mm/s", np.round(2*dv0, 4), "v_up in mm/s", np.round(v_1, 2), "del v_up in mm/s", np.round(dv_1,3), "v_down in mm/s", np.round(v_2,2), "del v_down in mm/s", np.round(dv_2,3),"reine Abweichung", np.round(f,2), "Abweichung varianzgewichtet", np.round(z, 2))

    n = 0
    nn = 0

    res_up = []
    res_dup = []
    res_down = []
    res_ddown = []

    for e, data in enumerate(z):
        if data < 3:
            n+=1
            #res_up += [v_1[e]]
            #res_down += [v_2[e]]


    for e,data in enumerate(f):
        if data < 0.317:
            nn+=1
            if v_1[e] < v_2[e]:
                res_up += [v_1[e]]
                res_dup += [dv_1[e]]
                res_down += [v_2[e]]
                res_ddown += [dv_2[e]]

    print(len(v_0))
    print(n)
    print(nn)
    print("----------")

    task_242_g([res_up,res_dup, res_down,res_ddown])

def surrounding_primes(n):
    
    def is_prime(num):
    
        if num < 2:
            return False
        for i in range(2, int(num**0.5) + 1):
            if num % i == 0:
                return False
        return True

    candidate_above = max(2, n)
    while not is_prime(candidate_above):
        candidate_above += 1

    candidate_below = n - 1
    while candidate_below > 1 and not is_prime(candidate_below):
        candidate_below -= 1

    if candidate_below < 2:
        candidate_below = None

    return [candidate_below, candidate_above]

def calculate_gcd(values, uncertainty):

    e = 1.6e-19
    
    avg_value = np.mean(values)
    theoretical_gcd = round(avg_value / e)
    
    scale_factor = 10**20  # Größenordnung anpassen
    scaled_values = np.round(values * (scale_factor)).astype(int)  # Skalierung
    
    print(" ")
    prime = surrounding_primes(theoretical_gcd)
    print(prime[0])
    print(theoretical_gcd)
    print(prime[1])
    print(" ")
    
    print(scaled_values)
    # GCD der nominalen Werte
    gcd_nominal = np.gcd.reduce(scaled_values)
    
    # Suche nach größeren GCDs um den theoretischen Wert
    gcd_candidates = []
    for test_gcd in range(prime[0] - 50, prime[1] + 50):  # Bereich um theor. GCD
        if all(np.isclose(val % test_gcd, 0, atol=1) for val in scaled_values):
            gcd_candidates.append(test_gcd)
    
    # Rückskalierung der GCDs
    gcd_candidates = [g for g in gcd_candidates]
    
    # Unsicherheitsbereich für GCD
    min_values_scaled = np.round((values - uncertainty) * (scale_factor)).astype(int)
    max_values_scaled = np.round((values + uncertainty) * (scale_factor * e)).astype(int)
    gcd_min = np.gcd.reduce(min_values_scaled) 
    gcd_max = np.gcd.reduce(max_values_scaled)

    # Rückgabe der Ergebnisse
    #return {
    #   "nominal": gcd_candidates[0] if gcd_candidates else gcd_nominal,
    #    "range": (gcd_min, gcd_max)
    #}

    return [prime[0] if abs(prime[0]-theoretical_gcd) < abs(prime[1]-theoretical_gcd) else prime[1], abs(prime[1] - prime[0])]


def task_242_g(data):
    eta = 18.19 * 10**-6
    deta = 0.003 * 18.19* 10**(-6)

    g = 9810 
    rho_ol = 886
    rho_luft = 1.225

    E = 512

    v_up = np.array(data[0])
    dv_up = np.array(data[1])
    v_down = np.array(data[2])
    dv_down = np.array(data[3])

    print(len(v_up))
    
    r = np.sqrt(9*eta*(v_down - v_up) / (4*g*(rho_ol - rho_luft))) * 10**6
    dr = math.sqrt(9/(4*g*(rho_ol - rho_luft)))*np.sqrt((deta*1/2*eta**(-0.5)*(v_down-v_up)*(v_down - v_up)**(-0.5))**2 
                                                        + (dv_down*eta/(2*(eta*(v_down - v_up))**0.5))**2
                                                        + (dv_up*eta/(2*(eta*(v_down - v_up))**0.5))**2) *3/4* 10**6

    ne = np.array(3*math.pi*eta*r*(v_up + v_down)*10**-3/E)*10**(-6)
    dne = 3*math.pi*np.sqrt((dv_up*eta*r/E)**2 + (dv_down*eta*r/E)**2
                            + (deta*r*(v_up + v_down)/E)**2 + (dr*eta/E*(v_down + v_up))**2
                            + (2*eta*r*(v_up + v_down)/E**2)**2)*10**(-9)*2

    n_cluster = 3
    ne_reshaped = ne.reshape(-1, 1)
    kmeans = KMeans(n_clusters=n_cluster, random_state=0).fit(ne_reshaped)
    labels = kmeans.labels_

    grouped_ne = [ne[labels == i] for i in range(n_cluster)]
    grouped_dne = [dne[labels == i] for i in range(n_cluster)]

    print(" ")

    e_val = 1.602176634*10**(-19)
    N = 1

    table_plot("Ohne Cunningham-Korrektur", "r in mikro m", np.round(r, 2), "dr in mikro m", np.round(dr,2),  "ne in As", np.round(ne, 18),"dne in As", np.round(dne, 19))

    fig, ax = plt.subplots()
    
    # GCD für jede Gruppe berechnen 
    diff = []
    e_S = []
    de_S = []
    gcdd = []
    dgcdd = []

    for i, group in enumerate(grouped_ne):
        gcd, dgcd = calculate_gcd(group, grouped_dne[i])
        
        rcentre = r[np.where(ne == group[0])]
        rtop = r[np.where(ne == group[-1])]

        ax.hlines(y=gcd*e_val, xmin= rcentre-0.1, xmax= rtop+0.1, linestyle="--", color="red", label="ggt value")
        dif = []
        for e in group:
            e_S.append(e/gcd)
            de_S.append(math.sqrt((dne[np.where(ne == e)]/gcd)**2 + (dgcd*e/gcd**2)**2))
            gcdd.append(gcd)
            dgcdd.append(dgcd)
            if e > gcd*e_val:
                dif.append(1 - gcd*e_val/e)
                #ax.vlines(x=r[np.where(ne == e)], ymin=gcd*e_val, ymax=e, linestyle="--", color="red")
            else:
                dif.append(1 - e/(gcd*e_val))
                #ax.vlines(x=r[np.where(ne == e)], ymin=e, ymax=gcd*e_val, linestyle="--", color="red")

        print(np.mean(dif))
        diff.append(np.mean(dif))

    #table_plot("Bestimmung von e_S,i", "e_S in C", e_S, "del e_S in C", de_S, "GGT", gcdd, "del GGT", dgcdd)

    print(" ")
    print(np.mean(diff))

    ax.grid()
    ax.errorbar(r, ne, xerr=dr, yerr=dne, fmt="x", label="values")
    ax.legend()
    ax.set_xlabel("r[mikro m]")
    ax.set_ylabel("Ne[C]")
    #plt.show()

    print(e_S)
    task_242_i(r, dr, e_S, de_S)
    

def task_242_i(r, dr1, e_S, de):
    fig, ax = plt.subplots()

    r1 = 1/np.array(r)
    dr = 1/(np.array(r)**2) * dr1

    y = np.array(e_S)**(2/3)
    dy = 2/3*np.array(e_S)**(-1/3)*de

    res = geradenfit(r1, y, dr, dy)

    print("Gerade")
    print(res["m"])
    print(res["dm"])
    print(res["b"])
    print(res["db"])
    print(1 - abs(res["r_squared"]))
    print()
    

    e_0 = res["b"]**(3/2)
    de_0 = 3/2*res["db"]**(1/2)*10**(-2)
   
    print("E_0") 
    print(e_0)
    print(de_0)

    e = 1.602176634 * 10**(-19)
    f = 1 - e_0/e

    print(f)

    print("H   ")
    print(em*10**(4))
    print(e_0)
    m = e_0 / (em*10**(4))
    dem = 1.7*10**10
    dm = math.sqrt((de_0/em)**2 + (dem*e_0/em**2)**2)

    m_0 = 9.109e-31

    print("M")
    print(m)
    print(m_0)
    print(dm)

    ax.grid()
    ax.errorbar(r1, y,xerr=dr, yerr=dy, fmt="x", label="values")
    ax.plot(r1, res["m"]*r1 + res["b"], label="fit")
    ax.set_xlabel(r"$\frac{1}{r}$[$(micro \text{ } m)^{-1}$]")
    ax.set_ylabel(r"$e_S^{2/3}[C]$")
    ax.legend()
    #plt.show()


task_242_c(data["c"], [data["d1"], data["d2"], data["d3"]])


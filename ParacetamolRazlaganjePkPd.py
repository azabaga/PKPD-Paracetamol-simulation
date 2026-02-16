# PK/PD analiza paracetamola bez generisanja "stvarnih" podataka
# - Čita tvoj CSV sa mjerenjima (ID, Dose_mg, Time_h, Conc_mg_L)
# - Simulira PK (1-kompartment, oralno; k_a & k_e prvog reda)
# - Računa greške simulacije u odnosu na CSV vrijednosti
# - Računa PD (Emax) efekat na osnovu simuliranih koncentracija
# - Prikazuje grafove (bez snimanja)

import os
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from dataclasses import dataclass

# ==== PODESI PUTANJU DO TVOG CSV FAJLA ====
# Npr: csv_path = r"C:\Users\Osman\Desktop\paracetamol_true.csv"
csv_path = os.path.join(os.getcwd(), "paracetamol_true.csv")

# ==== PK / PD MODELI ====
# “Koliko paracetamola ima u krvi u svakom trenutku, ako sam popio jednu tabletu, znajući koliko brzo se upija i koliko brzo ga tijelo izbacuje.”
def conc_oral_one_comp(t_h: np.ndarray, dose_mg: float, F: float, ka_h: float, ke_h: float, V_L: float) -> np.ndarray:
    """
    Oralni 1-kompartmentni PK sa apsorpcijom i eliminacijom 1. reda.
    Vraća koncentraciju (mg/L) za svaku t u t_h (h).
    """
    
    """
    t_h — vremena u satima (npr. [0.5, 1, 2, 4, 8, 12])

    dose_mg — doza lijeka (npr. 500 mg)

    F — bioiskoristivost (dio lijeka koji stvarno uđe u krv, npr. 0.9 = 90%)

    ka_h — brzina apsorpcije iz želuca u krv (1/sat)

    ke_h — brzina eliminacije iz tijela (1/sat)

    V_L — zapremina distribucije (koliko “prostora” ima lijek u tijelu, u litrima)
    """
    t_h = np.asarray(t_h, dtype=float)
    if abs(ka_h - ke_h) < 1e-8:
        C = (F * dose_mg / V_L) * (ka_h * t_h) * np.exp(-ka_h * t_h)
    else:
        C = (F * dose_mg / V_L) * (ka_h / (ka_h - ke_h)) * (np.exp(-ke_h * t_h) - np.exp(-ka_h * t_h)) # racuna koncentraciju lijeka u krvi
    C[C < 0] = 0.0
    return C

def pd_emax(conc_mg_L: np.ndarray, Emax: float, EC50_mg_L: float) -> np.ndarray:
    """Emax PD (0–100%): efekt kao funkcija koncentracije."""
    
    """
    ovo je PD (farmakodinamička) funkcija, tj. opisuje kakav efekat lijek ima na tijelo u zavisnosti od koncentracije u krvi.
    Ako je prethodna funkcija (conc_oral_one_comp) govorila koliko lijeka ima u krvi, ova govori koliko jako lijek djeluje pri toj koncentraciji.
    """
    
    """
    conc_mg_L → koncentracija lijeka u krvi (niz vrijednosti, mg/L)

    Emax → maksimalni mogući efekat (npr. 100% ublažavanja bola)

    EC50_mg_L → koncentracija pri kojoj se postiže 50% efekta
    """
    
    C = np.maximum(0.0, conc_mg_L)
    return Emax * C / (EC50_mg_L + C)

@dataclass
class PKParams:
    F: float
    ka: float
    ke: float
    V: float

@dataclass
class PDParams:
    Emax: float
    EC50: float

# ==== POPULACIONI PARAMETRI (ilustrativno, za acetaminofen) ====
# t1/2 ≈ 2.8 h → k_e ≈ ln(2)/t1/2 ≈ 0.25 1/h
pop_pk = PKParams(F=0.9, ka=1.5, ke=0.25, V=50.0)
pop_pd = PDParams(Emax=100.0, EC50=10.0)

# ==== UČITAJ CSV ====
required_cols = {"ID", "Dose_mg", "Time_h", "Conc_mg_L"}
df = pd.read_csv(csv_path)

missing = required_cols - set(df.columns)
if missing:
    raise ValueError(f"Nedostaju kolone u CSV: {missing}. Očekivano: {sorted(required_cols)}")

# Očisti tipove i sort
df["ID"] = df["ID"].astype(int)
df["Dose_mg"] = df["Dose_mg"].astype(float)
df["Time_h"] = df["Time_h"].astype(float)
df["Conc_mg_L"] = df["Conc_mg_L"].astype(float)
df = df.sort_values(["ID", "Time_h"]).reset_index(drop=True)

# ==== SIMULACIJA (koristeći populacione parametre) ====

"""
    Taj dio koda prolazi kroz svakog pacijenta i računa modelsku koncentraciju paracetamola u krvi u svakom vremenskom trenutku,
    pa spaja sve rezultate u jedan DataFrame (sim_df) koji sadrži “idealne” vrijednosti iz simulacije.
    
    To jest:
    Pogleda kolika je bila doza (npr. 500 mg).

    Uzme vremena mjerenja (npr. 0.5h, 1h, 2h, 4h...).

    Izračuna pomoću formule koliko bi lijek teoretski trebao biti u krvi u tim trenucima.
"""

sim_rows = []
for pid, g in df.groupby("ID"):
    dose = float(g["Dose_mg"].iloc[0])
    t = g["Time_h"].to_numpy(dtype=float)
    c_sim = conc_oral_one_comp(t, dose, pop_pk.F, pop_pk.ka, pop_pk.ke, pop_pk.V)
    sim_rows.append(pd.DataFrame({
        "ID": pid,
        "Dose_mg": dose,
        "Time_h": t,
        "Conc_model_mg_L": c_sim
    }))

sim_df = pd.concat(sim_rows, ignore_index=True).sort_values(["ID", "Time_h"]).reset_index(drop=True)

# ==== UPOREDI SA CSV (greške) ====
merged = pd.merge(df, sim_df, on=["ID", "Dose_mg", "Time_h"], how="inner")
merged["Abs_Error"] = np.abs(merged["Conc_model_mg_L"] - merged["Conc_mg_L"])
eps = 1e-8
merged["Pct_Error"] = np.where(merged["Conc_mg_L"] > eps,
                               100.0 * (merged["Conc_model_mg_L"] - merged["Conc_mg_L"]) / merged["Conc_mg_L"],
                               np.nan)

def rmse(x: np.ndarray) -> float:
    x = np.asarray(x, dtype=float)
    return float(np.sqrt(np.mean(np.square(x))))

metrics = []
for pid, g in merged.groupby("ID"):
    
    """
    MAE_mg_L	prosječna apsolutna greška (koliko mg/L model u prosjeku “promaši”)
    RMSE_mg_L	“teža” greška (kažnjava velika odstupanja više)
    Bias_mg_L	prosječno odstupanje (pozitivno → model precjenjuje, negativno → podcjenjuje)
    MAPE_percent	prosječna greška u procentima (%)
    """
    
    diff = g["Conc_model_mg_L"] - g["Conc_mg_L"]
    metrics.append({
        "ID": pid,
        "Dose_mg": float(g["Dose_mg"].iloc[0]),
        "MAE_mg_L": float(np.mean(np.abs(diff))),
        "RMSE_mg_L": rmse(diff.to_numpy()),
        "Bias_mg_L": float(np.mean(diff)),
        "MAPE_percent": float(np.nanmean(np.abs(g["Pct_Error"])))
    })
metrics_df = pd.DataFrame(metrics).sort_values("ID").reset_index(drop=True)

# ==== PD (Emax) iz SIMULISANIH koncentracija ====

"""
prelazi sa farmakokinetike (PK) na farmakodinamiku (PD),
odnosno: sada ne gleda koliko lijeka ima u krvi, nego koliko jak efekat lijek ima u svakom trenutku.
zove onu gore vec objasnjenu pd_emax funkciju
"""

pd_rows = []
for pid, g in sim_df.groupby("ID"):
    effect = pd_emax(g["Conc_model_mg_L"].to_numpy(), pop_pd.Emax, pop_pd.EC50)
    pd_rows.append(pd.DataFrame({
        "ID": pid,
        "Time_h": g["Time_h"].to_numpy(),
        "Effect_percent": effect
    }))
pd_df = pd.concat(pd_rows, ignore_index=True).sort_values(["ID", "Time_h"]).reset_index(drop=True)

# ==== GRAFOVI (PRIKAZ, NE SNAJU) ====
# PK: Stvarno (CSV) vs Model
for pid, g in merged.groupby("ID"):
    plt.figure()
    plt.plot(g["Time_h"], g["Conc_mg_L"], marker="o", linestyle="-", label="Stvarno (CSV)")
    plt.plot(g["Time_h"], g["Conc_model_mg_L"], marker="x", linestyle="--", label="Model (sim)")
    plt.title(f"Paracetamol PK — Pacijent {pid} (Doza {int(g['Dose_mg'].iloc[0])} mg)")
    plt.xlabel("Vrijeme (h)")
    plt.ylabel("Koncentracija (mg/L)")
    plt.legend()

# PD: Emax efekat iz modelskih koncentracija
for pid, g in pd_df.groupby("ID"):
    plt.figure()
    plt.plot(g["Time_h"], g["Effect_percent"], marker="o", linestyle="-")
    plt.title(f"Paracetamol PD (Emax) — Pacijent {pid}")
    plt.xlabel("Vrijeme (h)")
    plt.ylabel("Analgetski efekat (%)")

plt.show()

# ==== Brzi ispisi ====
print("\nMetrike greške (po pacijentu):")
print(metrics_df)

print("\nPrvih 10 redova uporedbe (CSV vs model):")
print(merged.head(10))

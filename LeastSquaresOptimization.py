# -*- coding: utf-8 -*-
# PK fit po pacijentu: prije vs. poslije optimizacije (Spyder-ready)

import os
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from scipy.optimize import least_squares

# ------------------------------------------------------------
# KONFIGURACIJA
# ------------------------------------------------------------
CSV_PATH = os.path.join(os.getcwd(), "paracetamol_true.csv")  # <-- promijeni!
patient_id = 3           # npr. 1; ako None, uzima prvog iz fajla
fit_Tlag = True           # uključi procjenu kašnjenja apsorpcije Tlag (h)
use_weighted_residuals = True  # proportional weights ~ 1/y
F = 0.85                  # fiksna bioraspoloživost (frakcija)
pop_ka, pop_ke, pop_V = 1.5, 0.2, 70.0  # populacioni parametri (prije optimizacije)

# Granice parametara (zdravi odrasli; prilagodi prema potrebi)
bounds_no_tlag = ([0.1, 0.05, 20.0], [5.0, 1.0, 120.0])         # ka, ke, V
bounds_with_tlag = ([0.1, 0.05, 20.0, 0.0], [5.0, 1.0, 120.0, 1.0])  # + Tlag (0–1 h)

# ------------------------------------------------------------
# MODEL: oralni 1-kompartmentni sa opcionalnim Tlag
# ------------------------------------------------------------
def conc_oral_one_comp_tlag(t, dose_mg, F, ka, ke, V, Tlag=0.0):
    """
    Analitičko rješenje 1-kompartmentnog PK sa apsorpcijom/eliminacijom 1. reda.
    t: vrijeme (h), dose_mg: doza (mg), F: bioraspoloživost, V: L, ka/ke: 1/h, Tlag: h
    Vraća koncentraciju (mg/L).
    """
    t = np.asarray(t, float)
    te = np.clip(t - Tlag, 0.0, None)  # efektivno vrijeme nakon kašnjenja apsorpcije
    same = np.isclose(ka, ke)
    C = np.empty_like(te, dtype=float)
    if np.any(same):
        C[same] = (F * dose_mg / V) * (ka * te[same]) * np.exp(-ka * te[same])
    C[~same] = (F * dose_mg / V) * (ka / (ka - ke)) * (np.exp(-ke * te[~same]) - np.exp(-ka * te[~same]))
    C[C < 0] = 0.0
    return C

# ------------------------------------------------------------
# METRIKE
# ------------------------------------------------------------
def mae(yhat, y):  return float(np.mean(np.abs(yhat - y)))
def rmse(yhat, y): return float(np.sqrt(np.mean((yhat - y)**2)))
def mape(yhat, y):
    y_safe = np.clip(np.asarray(y, float), 1e-8, None)
    return float(np.mean(np.abs((yhat - y) / y_safe)) * 100.0)
def bias(yhat, y): return float(np.mean(yhat - y))

# ------------------------------------------------------------
# FIT PO PACIJENTU
# ------------------------------------------------------------
def fit_patient(g, F=0.85, fit_Tlag=True, weighted=True,
                p0=(1.2, 0.2, 70.0, 0.2),
                bounds_with_tlag=([0.1, 0.05, 20.0, 0.0], [5.0, 1.0, 120.0, 1.0]),
                bounds_no_tlag=([0.1, 0.05, 20.0], [5.0, 1.0, 120.0])):
    """
    g: DataFrame sa kolonama Time_h, Conc_mg_L, Dose_mg (jedan ID)
    Vraća dict s procijenjenim parametrima i metrikama.
    """
    t = g["Time_h"].to_numpy(float)
    y = g["Conc_mg_L"].to_numpy(float)
    dose = float(g["Dose_mg"].iloc[0])

    if fit_Tlag:
        p0_fit = np.array([p0[0], p0[1], p0[2], p0[3]], dtype=float)  # ka, ke, V, Tlag
        lb, ub = bounds_with_tlag
    else:
        p0_fit = np.array([p0[0], p0[1], p0[2]], dtype=float)        # ka, ke, V
        lb, ub = bounds_no_tlag

    def resid(p):
        if fit_Tlag:
            ka, ke, V, Tlag = p
            yhat = conc_oral_one_comp_tlag(t, dose, F, ka, ke, V, Tlag)
        else:
            ka, ke, V = p
            yhat = conc_oral_one_comp_tlag(t, dose, F, ka, ke, V, 0.0)
        r = yhat - y
        if weighted:
            w = 1.0 / np.clip(y, 0.1, None)  # proporcionalno: veća težina pri manjim C
            r = r * w
        return r

    sol = least_squares(resid, p0_fit, bounds=(lb, ub), method="trf", xtol=1e-10, ftol=1e-10)
    p = sol.x

    if fit_Tlag:
        ka, ke, V, Tlag = float(p[0]), float(p[1]), float(p[2]), float(p[3])
    else:
        ka, ke, V, Tlag = float(p[0]), float(p[1]), float(p[2]), 0.0

    # metrika poslije fita
    yhat_after = conc_oral_one_comp_tlag(t, dose, F, ka, ke, V, Tlag)
    out = {
        "ka": ka, "ke": ke, "V": V, "Tlag": Tlag,
        "MAE": mae(yhat_after, y),
        "RMSE": rmse(yhat_after, y),
        "MAPE": mape(yhat_after, y),
        "Bias": bias(yhat_after, y),
        "yhat_after": yhat_after
    }
    return out

# ------------------------------------------------------------
# UČITAVANJE PODATAKA I ODABIR PACIJENTA
# ------------------------------------------------------------
df = pd.read_csv(CSV_PATH)
required_cols = {"ID", "Dose_mg", "Time_h", "Conc_mg_L"}
missing = required_cols - set(df.columns)
if missing:
    raise ValueError(f"Nedostaju kolone u CSV-u: {missing}. Očekujem kolone: {required_cols}")

if patient_id is None:
    patient_id = int(df["ID"].iloc[0])

g = df[df["ID"] == patient_id].copy().sort_values("Time_h")
if g.empty:
    raise ValueError(f"Nije pronađen pacijent sa ID={patient_id} u {CSV_PATH}")

dose_mg = float(g["Dose_mg"].iloc[0])
t = g["Time_h"].to_numpy(float)
y = g["Conc_mg_L"].to_numpy(float)

# ------------------------------------------------------------
# PRORAČUN: PRIJЕ I POSLIJE OPTIMIZACIJE
# ------------------------------------------------------------
y_before = conc_oral_one_comp_tlag(t, dose_mg, F, pop_ka, pop_ke, pop_V, 0.0)

fit_res = fit_patient(
    g, F=F, fit_Tlag=fit_Tlag, weighted=use_weighted_residuals,
    p0=(pop_ka, pop_ke, pop_V, 0.2),
    bounds_with_tlag=bounds_with_tlag,
    bounds_no_tlag=bounds_no_tlag
)

y_after = fit_res["yhat_after"]

# Metrike prije fita
MAE_b = mae(y_before, y)
RMSE_b = rmse(y_before, y)
MAPE_b = mape(y_before, y)
Bias_b = bias(y_before, y)

# ------------------------------------------------------------
# ISPIS REZULTATA
# ------------------------------------------------------------
print(f"Pacijent ID = {patient_id}, Doza = {dose_mg} mg, N tačaka = {len(g)}")
print("\n--- Prije optimizacije (populacioni parametri) ---")
print(f"ka={pop_ka:.3f}  ke={pop_ke:.3f}  V={pop_V:.1f} L  Tlag=0.00 h (fiksno)")
print(f"MAE={MAE_b:.3f} mg/L  RMSE={RMSE_b:.3f} mg/L  MAPE={MAPE_b:.1f}%  Bias={Bias_b:.3f} mg/L")

print("\n--- Nakon optimizacije (individualni fit) ---")
print(f"ka={fit_res['ka']:.3f}  ke={fit_res['ke']:.3f}  V={fit_res['V']:.1f} L  Tlag={fit_res['Tlag']:.2f} h")
print(f"MAE={fit_res['MAE']:.3f} mg/L  RMSE={fit_res['RMSE']:.3f} mg/L  MAPE={fit_res['MAPE']:.1f}%  Bias={fit_res['Bias']:.3f} mg/L")

# ------------------------------------------------------------
# GRAF: stvarni podaci vs. model prije/poslije optimizacije
# ------------------------------------------------------------
plt.figure(figsize=(8, 5))
plt.scatter(t, y, color="black", s=45, label="Stvarni podaci", zorder=3)
plt.plot(t, y_before, "--", color="royalblue", linewidth=2, label="Model prije optimizacije")
plt.plot(t, y_after, "-", color="crimson", linewidth=2, label="Model nakon optimizacije")
plt.title(f"Poređenje modela prije i nakon optimizacije (ID={patient_id})", fontsize=11)
plt.xlabel("Vrijeme (h)")
plt.ylabel("Koncentracija (mg/L)")
plt.legend()
plt.grid(True, linestyle="--", alpha=0.6)
plt.tight_layout()
plt.show()

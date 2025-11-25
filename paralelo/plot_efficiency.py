import matplotlib.pyplot as plt
import pandas as pd
import os
import math

# ======= CONFIGURACIÓN ==========
archivo_tiempos = "tiempos_omp16_32.txt"   # mismo archivo que usaste para speedup
output_dir = "eficiencia"             # carpeta donde se guardará la figura
os.makedirs(output_dir, exist_ok=True)

# ======= LEER LOS DATOS ==========
# Formato: Ncells  Tiempo  Hilos
df = pd.read_csv(archivo_tiempos,
                 delim_whitespace=True,
                 header=None,
                 names=["Ncells", "Tiempo", "Hilos"])

# Opcional: limitar a 16 hilos
df = df[df["Hilos"] <= 32]

# ======= PROMEDIO POR (Ncells, Hilos) ==========
df_avg = df.groupby(["Ncells", "Hilos"])["Tiempo"].mean().reset_index()

# ======= BASE SECUENCIAL T_s(N) (1 hilo) ==========
Tseq = df_avg[df_avg["Hilos"] == 1].set_index("Ncells")["Tiempo"]
# Tseq[64], Tseq[128], etc.

# ======= CALCULAR SPEEDUP Y EFICIENCIA ==========
df_eff = df_avg.copy()

def calc_speedup(row):
    n  = row["Ncells"]
    tp = row["Tiempo"]
    ts = Tseq.get(n, float("nan"))
    return ts / tp

df_eff["Speedup"]    = df_eff.apply(calc_speedup, axis=1)
df_eff["Efficiency"] = df_eff["Speedup"] / df_eff["Hilos"]

# ======= GRAFICAR TODO EN UNA FIGURA ==========
plt.figure(figsize=(10, 6))

colores = ["#1f77b4", "#ff7f0e", "#2ca02c", "#d62728",
           "#9467bd", "#8c564b", "#e377c2", "#7f7f7f", "#bcbd22", "#17becf"]

N_unicos = sorted(df_eff["Ncells"].unique())
max_eff = 0.0

for i, n in enumerate(N_unicos):
    datos = df_eff[df_eff["Ncells"] == n].sort_values("Hilos")
    plt.plot(datos["Hilos"], datos["Efficiency"],
             marker='o',
             label=f"N_celdas = {n}",
             color=colores[i % len(colores)])
    max_eff = max(max_eff, datos["Efficiency"].max())

# Línea de eficiencia ideal (0.7)
plt.axhline(y=0.7, linestyle='--', color='black', label='Eficiencia ideal (0.7)')


# Estética
plt.xlabel("Número de hilos", fontsize=15)
plt.ylabel("Eficiencia $E = \\frac{T_s}{p\\,T_p}$", fontsize=15)
plt.title("Eficiencia vs número de hilos\n(Marching Cubes paralelo)", fontsize=16.5)
plt.grid(True)
plt.legend(fontsize=12)
plt.tick_params(axis='both', which='major', labelsize=12)

# Límite en Y (con un poco de margen arriba, pero máximo ~1.1)
plt.ylim(0, min(1.1, math.ceil(max_eff * 10) / 10))

plt.tight_layout()
salida = os.path.join(output_dir, "efiencia_mc.png")
plt.savefig(salida)
plt.close()

print(f"Gráfica de eficiencia guardada en: {salida}")

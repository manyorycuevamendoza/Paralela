import matplotlib.pyplot as plt
import pandas as pd
import os
import math

# ======= CONFIGURACIÓN ==========
archivo_tiempos = "tiempos_omp16_32.txt"   # tu archivo con Ncells, tiempo, hilos
output_dir = "speedup"             # carpeta donde se guardarán las figuras
os.makedirs(output_dir, exist_ok=True)

# ======= LEER LOS DATOS ==========
# Formato: Ncells time threads
df = pd.read_csv(archivo_tiempos,
                 delim_whitespace=True,
                 header=None,
                 names=["Ncells", "Tiempo", "Hilos"])

# Si quieres limitar a 16 hilos (por si en Khipu probaste más)
df = df[df["Hilos"] <= 32]

# ======= PROMEDIO POR (Ncells, Hilos) ==========
df_prom = df.groupby(["Ncells", "Hilos"])["Tiempo"].mean().reset_index()
print("Datos promediados por (Ncells, Hilos):")
print(df_prom)


# ======= BASE SECUENCIAL (T_seq PARA CADA Ncells) ==========
Tseq = df_prom[df_prom["Hilos"] == 1].set_index("Ncells")["Tiempo"]
# Tseq es una serie: índice = Ncells, valor = tiempo con 1 hilo
print("Tiempos secuenciales (1 hilo):")
print(Tseq)


# ======= CALCULAR SPEEDUP: S_p = Tseq / Tp ==========
df_speedup = df_prom.copy()

def calc_speedup(row):
    n  = row["Ncells"]
    tp = row["Tiempo"]
    ts = Tseq.get(n, float("nan"))
    return ts / tp

df_speedup["Speedup"] = df_speedup.apply(calc_speedup, axis=1)
print("Datos con speedup calculado:")
print(df_speedup)




# ======= GRAFICAR TODO EN UNA SOLA FIGURA ==========
plt.figure(figsize=(10, 6))

colores = ["#1f77b4", "#ff7f0e", "#2ca02c", "#d62728",
           "#9467bd", "#8c564b", "#e377c2", "#7f7f7f"]

N_unicos = sorted(df_speedup["Ncells"].unique())
max_speedup = 0.0

for i, n in enumerate(N_unicos):
    datos = df_speedup[df_speedup["Ncells"] == n].sort_values("Hilos")
    plt.plot(datos["Hilos"], datos["Speedup"],
             marker='o',
             label=f"N_celdas = {n}",
             color=colores[i % len(colores)])
    max_speedup = max(max_speedup, datos["Speedup"].max())

# Línea de speedup ideal S_p = p
hilos_unicos = sorted(df_speedup["Hilos"].unique())
plt.plot(hilos_unicos, hilos_unicos,
         linestyle='--', color='black', label='Speedup ideal')

# Estética
plt.xlabel("Número de hilos", fontsize=15)
plt.ylabel("Speedup $T_s / T_p$", fontsize=15)
plt.title("Speedup vs número de hilos\n(Marching Cubes paralelo)", fontsize=16.5)
plt.grid(True)
plt.legend(fontsize=12)
plt.tick_params(axis='both', which='major', labelsize=12)

# Límite en Y con un pequeño margen
plt.ylim(0, math.ceil(max_speedup + 0.5))

plt.tight_layout()
salida = os.path.join(output_dir, "speedup_mc32.png")
plt.savefig(salida)
plt.close()

print(f"Gráfica guardada en: {salida}")

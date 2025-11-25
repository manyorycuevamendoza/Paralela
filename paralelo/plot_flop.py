import matplotlib.pyplot as plt
import pandas as pd
import os

# ======= CONFIGURACIÓN ==========
# Ruta del archivo que generó tu programa C++:
# ./mc_omp.exe > paralelo/tiempos_omp_flops.txt
archivo_tiempos = "paralelo/tiempos_omp_flops.txt"

# Carpeta donde se guardarán las figuras
output_dir = "flops"
os.makedirs(output_dir, exist_ok=True)

# ======= LEER LOS DATOS ==========
# Formato de cada línea:
# Ncells   Tiempo(s)   Hilos   FLOPs   GFLOPS
df = pd.read_csv(
    archivo_tiempos,
    delim_whitespace=True,
    header=None,
    names=["Ncells", "Tiempo", "Hilos", "FLOPs", "GFLOPS"]
)

# ======= PROMEDIAR POR (Ncells, Hilos) ==========
df_avg = df.groupby(["Ncells", "Hilos"]).mean().reset_index()

print("Primeras filas promediadas:")
print(df_avg.head())

# ------------------------------------------------------------------
# 1) GRÁFICA: GFLOPS vs #hilos para varios tamaños Ncells
# ------------------------------------------------------------------
plt.figure(figsize=(10, 6))

# Puedes elegir qué Ncells graficar.
# Aquí tomo solo los más grandes (>= 64) para que se vea mejor,
# pero puedes cambiar esta línea como quieras.
N_unicos = sorted(df_avg["Ncells"].unique())
N_a_graficar = [n for n in N_unicos if n >= 64]
if not N_a_graficar:   # por si solo tienes pequeños
    N_a_graficar = N_unicos

colores = [
    "#1f77b4", "#ff7f0e", "#2ca02c", "#d62728",
    "#9467bd", "#8c564b", "#e377c2", "#7f7f7f"
]

for i, n in enumerate(N_a_graficar):
    sub = df_avg[df_avg["Ncells"] == n].sort_values("Hilos")
    plt.plot(
        sub["Hilos"], sub["GFLOPS"],
        marker="o",
        label=f"N_celdas = {n}",
        color=colores[i % len(colores)]
    )

plt.xlabel("Número de hilos", fontsize=14)
plt.ylabel("GFLOPS", fontsize=14)
plt.title("GFLOPS vs número de hilos\n(Marching Cubes paralelo)", fontsize=16)
plt.grid(True)
plt.legend(fontsize=11)
plt.tick_params(axis='both', which='major', labelsize=12)
plt.tight_layout()

salida_gflops = os.path.join(output_dir, "gflops_vs_threads.png")
plt.savefig(salida_gflops)
plt.close()
print(f"Gráfica GFLOPS guardada en: {salida_gflops}")

# ------------------------------------------------------------------
# 2) (OPCIONAL) GRÁFICA: FLOPs totales vs #hilos
# ------------------------------------------------------------------
plt.figure(figsize=(10, 6))

for i, n in enumerate(N_a_graficar):
    sub = df_avg[df_avg["Ncells"] == n].sort_values("Hilos")
    plt.plot(
        sub["Hilos"], sub["FLOPs"],
        marker="o",
        label=f"N_celdas = {n}",
        color=colores[i % len(colores)]
    )

plt.xlabel("Número de hilos", fontsize=14)
plt.ylabel("FLOPs totales", fontsize=14)
plt.title("FLOPs vs número de hilos\n(Marching Cubes paralelo)", fontsize=16)
plt.grid(True)
plt.legend(fontsize=11)
plt.tick_params(axis='both', which='major', labelsize=12)
plt.tight_layout()

salida_flops = os.path.join(output_dir, "flops_vs_threads.png")
plt.savefig(salida_flops)
plt.close()
print(f"Gráfica FLOPs guardada en: {salida_flops}")

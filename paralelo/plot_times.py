import matplotlib.pyplot as plt
import pandas as pd

# ======= CONFIGURACIÓN ==========
archivo_tiempos = "tiempos_omp16_32.txt"
nombre_salida_png = "./tiempos/omp_time_vs_threads32.png"
titulo = "Tiempo medio vs número de hilos\n(Marching Cubes paralelo)"
etiqueta_x = "Número de hilos"
etiqueta_y = "Tiempo medio de ejecución (s)"
# ================================

# Leer datos: 3 columnas -> N_celdas, tiempo, hilos
df = pd.read_csv(
    archivo_tiempos,
    delim_whitespace=True,
    header=None,
    names=["N_celdas", "Tiempo", "Hilos"]
)

# Promediar por (N_celdas, Hilos)
df_prom = df.groupby(["N_celdas", "Hilos"])["Tiempo"].mean().reset_index()

# Crear figura
plt.figure(figsize=(10, 6))

# Una curva por cada N_celdas
for N in sorted(df_prom["N_celdas"].unique()):
    sub = df_prom[df_prom["N_celdas"] == N].sort_values("Hilos")
    plt.plot(
        sub["Hilos"],
        sub["Tiempo"],
        marker='o',
        linestyle='-',
        label=f"N_celdas = {N}"
    )

plt.title(titulo, fontsize=16)
plt.xlabel(etiqueta_x, fontsize=14)
plt.ylabel(etiqueta_y, fontsize=14)
plt.grid(True)
plt.legend(fontsize=12)
plt.tick_params(axis='both', which='major', labelsize=12)
plt.yscale("log")   # opcional: escala log en Y como en los ejemplos

plt.tight_layout()
plt.savefig(nombre_salida_png)
plt.close()

print(f"{nombre_salida_png} guardado.")

import matplotlib.pyplot as plt
import pandas as pd

# ======= CONFIGURACIÓN ==========
archivo_tiempos = "tiempos_sec.txt"          # archivo con tus tiempos secuenciales
nombre_salida_png = "time_seq_mc.png"        # imagen de salida
titulo = "Tiempo medio vs tamaño del problema (Marching Cubes secuencial)"
etiqueta_x = "Número de celdas por lado (N_celdas)"
etiqueta_y = "Tiempo medio de ejecución (s)"
# ================================

# Leer los datos: columnas = N_celdas, Tiempo, Procesos
df = pd.read_csv(
    archivo_tiempos,
    delim_whitespace=True,
    header=None,
    names=["N_celdas", "Tiempo", "Procesos"]
)

# Agrupar por N_celdas y calcular el promedio del tiempo
df_prom = df.groupby("N_celdas")["Tiempo"].mean().reset_index()

print("Promedios:")
print(df_prom)

fontsize_labels = 15
fontsize_title = 16.5

# Crear la gráfica
plt.figure(figsize=(10, 6))
plt.plot(
    df_prom["N_celdas"],
    df_prom["Tiempo"],
    marker='o',
    linestyle='-',
    label="Marching Cubes secuencial"
)

plt.title(titulo, fontsize=fontsize_title)
plt.xlabel(etiqueta_x, fontsize=fontsize_labels)
plt.ylabel(etiqueta_y, fontsize=fontsize_labels)
plt.grid(True)

plt.tick_params(axis='both', which='major', labelsize=12)
plt.legend(fontsize=14)

# Como los tiempos crecen bastante, la escala log en Y ayuda a ver mejor
plt.yscale("log")

# Guardar la gráfica
plt.savefig(nombre_salida_png, bbox_inches="tight")
plt.close()

print(f"{nombre_salida_png} guardado correctamente")

import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from scipy import stats

# Configuración visual y semilla para reproducibilidad
sns.set_theme(style="whitegrid")
np.random.seed(42)

# ==========================================
# REQUERIMIENTO 1: Carga y Exploración de Datos
# ==========================================
n = 200

# Simulación del DataFrame con las 4 variables requeridas
df = pd.DataFrame({
    "edad": np.random.randint(18, 45, size=n),
    "genero": np.random.choice(["Femenino", "Masculino", "Otro"], size=n, p=[0.48, 0.48, 0.04]),
    "satisfaccion": np.clip(np.round(np.random.normal(7.2, 1.4, size=n), 1), 1, 10),
    "horas_estudio": np.round(np.random.normal(10, 3, size=n), 1).clip(0, None),
})

# Inducción controlada de nulos para evidenciar su detección y tratamiento
idx_nulos = np.random.choice(df.index, size=5, replace=False)
df.loc[idx_nulos, "horas_estudio"] = np.nan

print("=== REQUERIMIENTO 1: EXPLORACIÓN DE DATOS ===")
print("Dimensiones del dataset:", df.shape)
print("\nPrimeros registros:")
print(df.head())

# Estadísticas descriptivas de variables numéricas
desc = df[["edad", "satisfaccion", "horas_estudio"]].describe().T[["mean", "50%", "std", "min", "max"]]
desc.columns = ["media", "mediana", "desv_std", "min", "max"]
print("\nEstadísticas Descriptivas:")
print(desc.round(2))

# Verificación de nulos
print("\nValores nulos por columna (inicial):")
print(df.isnull().sum())

# Tratamiento de valores nulos mediante imputación por mediana
df["horas_estudio"] = df["horas_estudio"].fillna(df["horas_estudio"].median())
print("\nValores nulos tras imputación con mediana:", df.isnull().sum().sum())


# ==========================================
# REQUERIMIENTO 2: Distribución y Visualización
# ==========================================
print("\n=== REQUERIMIENTO 2: DISTRIBUCIÓN Y VISUALIZACIÓN ===")
media_np = np.mean(df["satisfaccion"])
varianza_np = np.var(df["satisfaccion"], ddof=1)  # Varianza muestral

print(f"Media (NumPy):    {media_np:.4f}")
print(f"Varianza (NumPy): {varianza_np:.4f}")

# Gráfico de distribución
plt.figure(figsize=(8, 5))
sns.histplot(df["satisfaccion"], bins=15, kde=True, color="#6DA544")
plt.axvline(media_np, color="red", linestyle="--", label=f"Media = {media_np:.2f}")
plt.title("Histograma de Puntajes de Satisfacción")
plt.xlabel("Puntaje de Satisfacción (1-10)")
plt.ylabel("Frecuencia")
plt.legend()
plt.tight_layout()
plt.savefig("histograma_satisfaccion.png")
plt.show()

# Verificación de normalidad (Prueba Shapiro-Wilk)
shapiro_stat, shapiro_p = stats.shapiro(df["satisfaccion"])
print(f"Prueba Shapiro-Wilk: W = {shapiro_stat:.4f}, p-value = {shapiro_p:.4f}")


# ==========================================
# REQUERIMIENTO 3: Intervalo de Confianza (95%)
# ==========================================
print("\n=== REQUERIMIENTO 3: INTERVALO DE CONFIANZA ===")
media = df["satisfaccion"].mean()
s = df["satisfaccion"].std(ddof=1)
t_crit = stats.t.ppf(0.975, df=n-1)

margen_error = t_crit * (s / np.sqrt(n))
ic_inferior = media - margen_error
ic_superior = media + margen_error

print(f"Media Muestral: {media:.4f}")
print(f"Desviación Estándar (s): {s:.4f}")
print(f"Valor Crítico t (df=199): {t_crit:.4f}")
print(f"Margen de Error: {margen_error:.4f}")
print(f"IC 95%: [{ic_inferior:.4f}, {ic_superior:.4f}]")


# ==========================================
# REQUERIMIENTO 4: Prueba de Hipótesis
# ==========================================
print("\n=== REQUERIMIENTO 4: PRUEBA DE HIPÓTESIS ===")
mu_0 = 7.0
alpha = 0.05

t_stat, p_value = stats.ttest_1samp(df["satisfaccion"], popmean=mu_0)

print(f"Hipótesis Nula (H0): mu = {mu_0}")
print(f"Hipótesis Alternativa (H1): mu != {mu_0}")
print(f"Estadístico t: {t_stat:.4f}")
print(f"Valor-p: {p_value:.6f}")

if p_value < alpha:
    print(f"Conclusión: Se RECHAZA H0 (p-value < {alpha}). La media difiere significativamente de 7.")
else:
    print(f"Conclusión: NO se rechaza H0 (p-value >= {alpha}). No hay evidencia suficiente.")
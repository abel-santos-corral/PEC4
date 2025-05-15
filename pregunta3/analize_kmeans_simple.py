import numpy as np
import matplotlib.pyplot as plt
from sklearn.datasets import make_blobs
from sklearn.cluster import KMeans
from sklearn.metrics import adjusted_rand_score, normalized_mutual_info_score, homogeneity_score
import os

# Crear directorio si no existe
output_dir = "pregunta3/imagenes/kmeans"
os.makedirs(output_dir, exist_ok=True)

# Crear datos sintéticos con más solapamiento
X, y_true = make_blobs(n_samples=150, centers=3, cluster_std=1.9, random_state=42)

# Aplicar KMeans con k=3
kmeans = KMeans(n_clusters=3, random_state=42, n_init='auto')
y_pred = kmeans.fit_predict(X)

# Métricas
inercia = kmeans.inertia_
ari = adjusted_rand_score(y_true, y_pred)
nmi = normalized_mutual_info_score(y_true, y_pred)
homogeneidad = homogeneity_score(y_true, y_pred)

# Mostrar resultados
print("Resultados del clustering con KMeans (k=3):")
print(f"Inercia: {inercia:.2f}")
print(f"ARI: {ari:.3f}")
print(f"NMI: {nmi:.3f}")
print(f"Homogeneidad: {homogeneidad:.3f}")

# Graficar
plt.figure(figsize=(8, 6))
plt.scatter(X[:, 0], X[:, 1], c=y_pred, cmap='viridis', s=50)
centroids = kmeans.cluster_centers_
plt.scatter(centroids[:, 0], centroids[:, 1], c='red', s=200, alpha=0.6, marker='X', label='Centroides')
plt.title('Clustering KMeans con k=3 (dispersión media)')
plt.xlabel('Feature 1')
plt.ylabel('Feature 2')
plt.legend()
plt.grid(True)

# Guardar gráfico
output_path = os.path.join(output_dir, "kmeans_150_k3_mas_disperso.png")
plt.savefig(output_path)
plt.close()

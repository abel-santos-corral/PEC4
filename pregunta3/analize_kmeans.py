import os
import shutil
import numpy as np
import matplotlib.pyplot as plt
from sklearn.datasets import load_iris
from sklearn.cluster import KMeans
from sklearn.metrics import adjusted_rand_score, normalized_mutual_info_score, homogeneity_score

# Mensaje inicial
print("\033[34mEjecutando...\033[0m \033[33manalize_kmeans (clustering con doble leyenda)\033[0m")

# Crear carpeta si no existe, y borrar solo los archivos .png previos
output_dir = "pregunta3/imagenes/kmeans"
os.makedirs(output_dir, exist_ok=True)
for filename in os.listdir(output_dir):
    if filename.endswith(".png"):
        os.remove(os.path.join(output_dir, filename))

# Cargar el dataset Iris
iris = load_iris()
X = iris.data
y = iris.target
class_names = iris.target_names

# Usaremos las dos primeras features para graficar
X_plot = X[:, :2]
feature_names = iris.feature_names[:2]

# Configuraciones
sample_sizes = [10, 50, 150]
k_values = [1, 2, 3, 5]

# Formas y colores para clusters y clases
markers = ['o', 's', 'D', '^', 'v', 'P', '*']
colors = ['tab:blue', 'tab:orange', 'tab:green']

for sample_size in sample_sizes:
    indices = np.random.choice(len(X), size=sample_size, replace=False)
    X_sample = X[indices]
    X_sample_plot = X_plot[indices]
    y_sample = y[indices]

    for k in k_values:
        kmeans = KMeans(n_clusters=k, n_init="auto", random_state=42)
        kmeans.fit(X_sample)
        labels = kmeans.labels_
        centroids = kmeans.cluster_centers_[:, :2]

        # Métricas
        ari = adjusted_rand_score(y_sample, labels)
        nmi = normalized_mutual_info_score(y_sample, labels)
        homo = homogeneity_score(y_sample, labels)

        print(f"\n=== Muestra: {sample_size} | k = {k} ===")
        print(f"Inercia: {kmeans.inertia_:.2f}")
        print(f"ARI: {ari:.2f} | NMI: {nmi:.2f} | Homogeneidad: {homo:.2f}")

        # Crear figura más ancha
        fig, ax = plt.subplots(figsize=(9, 5))

        # Graficar cada punto con su color (clase real) y forma (cluster asignado)
        for i in range(sample_size):
            cluster_id = labels[i]
            class_id = y_sample[i]
            ax.scatter(
                X_sample_plot[i, 0], X_sample_plot[i, 1],
                c=colors[class_id],
                marker=markers[cluster_id % len(markers)],
                edgecolor='k',
                s=80,
                label=f"{cluster_id}-{class_id}"  # temporal para leyendas separadas
            )

        # Graficar centroides
        ax.scatter(
            centroids[:, 0], centroids[:, 1],
            s=200, marker='X', c='red', label='Centroides'
        )

        # Leyenda 1: por forma (clusters)
        cluster_legend = [
            plt.Line2D([0], [0], marker=markers[i % len(markers)], color='w',
                       label=f'Cluster {i}', markerfacecolor='gray', markeredgecolor='k', markersize=10)
            for i in range(k)
        ]

        # Leyenda 2: por color (clases reales)
        class_legend = [
            plt.Line2D([0], [0], marker='o', color='w',
                       label=class_names[i], markerfacecolor=colors[i], markeredgecolor='k', markersize=10)
            for i in range(len(class_names))
        ]

        ax.set_title(f'K-Means en Iris (2D)\nMuestra={sample_size}, k={k}')
        ax.set_xlabel(feature_names[0])
        ax.set_ylabel(feature_names[1])
        ax.grid(True)

        # Mostrar leyendas separadas en una columna al costado
        leg1 = ax.legend(handles=cluster_legend, title='Clusters (formas)', loc='upper left', bbox_to_anchor=(1.02, 1))
        leg2 = ax.legend(handles=class_legend, title='Clases reales (colores)', loc='lower left', bbox_to_anchor=(1.02, 0))
        ax.add_artist(leg1)

        plt.tight_layout()
        plt.savefig(f"{output_dir}/iris_muestra_{sample_size}_k_{k}.png", bbox_inches='tight')
        plt.close()

# Mensaje final
print("\033[34mFinalizado...\033[0m \033[33manalize_kmeans\033[0m")

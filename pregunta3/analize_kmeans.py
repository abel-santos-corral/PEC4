import numpy as np
import matplotlib.pyplot as plt
import os
from collections import Counter
from sklearn.datasets import make_blobs
from sklearn.neighbors import KNeighborsClassifier
from sklearn.model_selection import train_test_split
from sklearn.metrics import accuracy_score, confusion_matrix, f1_score

# Mensaje inicial
print("\033[34mEjecutando...\033[0m \033[33manalize_kmeans\033[0m")

# Crear carpeta de imágenes
output_dir = "pregunta3/imagenes/kmeans"
os.makedirs(output_dir, exist_ok=True)

# Crear datos sintéticos con 6 clústeres
X, y = make_blobs(n_samples=300, centers=6, cluster_std=1.2, random_state=42)

# Nuevo ejemplo para clasificar
nuevo_ejemplo = np.array([[0.0, 5.0]])

# Configuraciones
sample_sizes = [10, 50, 100]
k_values = [1, 2, 5]

for sample_size in sample_sizes:
    # Muestreo aleatorio
    indices = np.random.choice(len(X), size=sample_size, replace=False)
    X_sample = X[indices]
    y_sample = y[indices]

    # Verificar si se puede estratificar
    y_counts = Counter(y_sample)
    can_stratify = all(count >= 2 for count in y_counts.values())
    stratify_param = y_sample if can_stratify else None
    if not can_stratify:
        print(f"⚠️ No se puede estratificar muestra de tamaño {sample_size}. Split aleatorio usado.")

    # Split train/test
    X_train, X_test, y_train, y_test = train_test_split(
        X_sample, y_sample, test_size=0.3, stratify=stratify_param, random_state=42
    )

    for k in k_values:
        kmeans = KNeighborsClassifier(n_neighbors=k)
        kmeans.fit(X_train, y_train)

        y_pred = kmeans.predict(X_test)
        acc = accuracy_score(y_test, y_pred)
        f1 = f1_score(y_test, y_pred, average='macro')
        cm = confusion_matrix(y_test, y_pred)

        # Vecinos más cercanos del nuevo ejemplo
        distancias, indices_vecinos = kmeans.kneighbors(nuevo_ejemplo)
        clases_vecinos = y_train[indices_vecinos[0]]
        clases_vecinos_nombres = [f"clase {c}" for c in clases_vecinos]

        pred_nuevo = kmeans.predict(nuevo_ejemplo)[0]

        print(f"\n=== Muestra: {sample_size} | k = {k} ===")
        print(f"Precisión: {acc:.2f}")
        print(f"F1-score macro: {f1:.2f}")
        print("Matriz de confusión:")
        print(cm)
        print(f"Vecinos más cercanos (k={k}): {clases_vecinos_nombres}")
        print(f"→ Predicción del nuevo ejemplo: clase {pred_nuevo}")

        # Gráfico
        plt.figure(figsize=(6, 5))
        for class_index in np.unique(y_sample):
            plt.scatter(
                X_sample[y_sample == class_index, 0],
                X_sample[y_sample == class_index, 1],
                label=f"Clase {class_index}",
                marker='o',
                edgecolor='k'
            )
        plt.scatter(
            nuevo_ejemplo[0, 0], nuevo_ejemplo[0, 1],
            color='red', marker='*', s=200, label='Nuevo ejemplo'
        )
        plt.title(f'Clasificación con kmeans\nMuestra={sample_size}, k={k}')
        plt.xlabel('Feature 1')
        plt.ylabel('Feature 2')
        plt.grid(True)
        plt.legend(fontsize='small', loc='best')
        plt.tight_layout()
        plt.savefig(f"{output_dir}/muestra_{sample_size}_k_{k}.png")
        plt.close()

# Mensaje final
print("\033[34mFinalizado...\033[0m \033[33manalize_kmeans\033[0m")
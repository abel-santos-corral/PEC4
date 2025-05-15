import numpy as np
import matplotlib.pyplot as plt
import os
from collections import Counter
from sklearn.datasets import load_iris
from sklearn.neighbors import KNeighborsClassifier
from sklearn.model_selection import train_test_split
from sklearn.metrics import accuracy_score, confusion_matrix, f1_score

# Imprimir feedback por pantalla:
print("\033[34mEjecutando...\033[0m \033[33manalize_kmeans\033[0m")

# Crear carpetas necesarias
os.makedirs("pregunta3/imagenes/kmeans", exist_ok=True)

# Cargar datos iris
iris = load_iris()
X = iris.data
y = iris.target
target_names = iris.target_names

# Nuevo ejemplo corregido
nuevo_ejemplo = np.array([[6.2, 3.3, 4.8, 1.75]])

# Configuraciones
sample_sizes = [10, 50, 150]
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
        knn = KNeighborsClassifier(n_neighbors=k)
        knn.fit(X_train, y_train)

        y_pred = knn.predict(X_test)
        acc = accuracy_score(y_test, y_pred)
        f1 = f1_score(y_test, y_pred, average='macro')
        cm = confusion_matrix(y_test, y_pred)

        # Vecinos más cercanos del nuevo ejemplo
        distancias, indices_vecinos = knn.kneighbors(nuevo_ejemplo)
        clases_vecinos = y_train[indices_vecinos[0]]
        clases_vecinos_nombres = [str(target_names[c]) for c in clases_vecinos]

        pred_nuevo = knn.predict(nuevo_ejemplo)[0]
        pred_label = target_names[pred_nuevo]

        print(f"\n=== Muestra: {sample_size} | k = {k} ===")
        print(f"Precisión: {acc:.2f}")
        print(f"F1-score macro: {f1:.2f}")
        print("Matriz de confusión:")
        print(cm)
        print(f"Vecinos más cercanos (k={k}): {clases_vecinos_nombres}")
        print(f"→ Predicción del nuevo ejemplo: {pred_label} (clase {pred_nuevo})")

        # Gráfico 1: Petal length vs Petal width
        plt.figure(figsize=(6, 5))
        for class_index, class_label in enumerate(target_names):
            plt.scatter(
                X_sample[y_sample == class_index, 2],
                X_sample[y_sample == class_index, 3],
                label=f"Clase real: {class_label}",
                marker='o',
                edgecolor='k'
            )
        plt.scatter(
            nuevo_ejemplo[0, 2], nuevo_ejemplo[0, 3],
            color='red', marker='*', s=200, label='Nuevo ejemplo'
        )
        plt.title(f'Petal length vs Petal width\nMuestra={sample_size}, k={k}\nAcc={acc:.2f}, F1={f1:.2f}\nPred: {pred_label}')
        plt.xlabel('Petal length (cm)')
        plt.ylabel('Petal width (cm)')
        plt.grid(True)
        plt.legend(fontsize='small', loc='best')
        plt.tight_layout()
        plt.savefig(f"pregunta3/imagenes/kmeans/muestra_{sample_size}_k_{k}_petal.png")
        plt.close()

        # Gráfico 2: Sepal length vs Sepal width
        plt.figure(figsize=(6, 5))
        for class_index, class_label in enumerate(target_names):
            plt.scatter(
                X_sample[y_sample == class_index, 0],
                X_sample[y_sample == class_index, 1],
                label=f"Clase real: {class_label}",
                marker='o',
                edgecolor='k'
            )
        plt.scatter(
            nuevo_ejemplo[0, 0], nuevo_ejemplo[0, 1],
            color='red', marker='*', s=200, label='Nuevo ejemplo'
        )
        plt.title(f'Sepal length vs Sepal width\nMuestra={sample_size}, k={k}\nAcc={acc:.2f}, F1={f1:.2f}\nPred: {pred_label}')
        plt.xlabel('Sepal length (cm)')
        plt.ylabel('Sepal width (cm)')
        plt.grid(True)
        plt.legend(fontsize='small', loc='best')
        plt.tight_layout()
        plt.savefig(f"pregunta3/imagenes/kmeans/muestra_{sample_size}_k_{k}_sepal.png")
        plt.close()

# Imprimir feedback por pantalla:
print("\033[34mFinalizado...\033[0m \033[33manalize_kmeans\033[0m")

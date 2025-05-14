import pandas as pd

# Definición de la función heaviside
def heaviside(u):
    """Función escalón de Heaviside que retorna 1 si x >= 0, si no retorna 0."""
    return 1 if u > 0 else 0

# Imprimir feedback por pantalla:
print("\033[34mEjecutando...\033[0m \033[33mcalculo_red_neuronal\033[0m")

# Pesos de la red
# Filas: entradas x, y; columna z: pesos de salida (fila 'z')
W = {
    'h0': {'x': 2, 'y': 1},
    'h1': {'x': 0, 'y': -2},
    'h2': {'x': 1, 'y': 1}
}
W_out = {'h0': 1, 'h1': -1, 'h2': 0}

# Conjunto de datos de entrada
data = [
    {'x': 1,  'y': -1},
    {'x': 2,  'y': -2},
    {'x': -1, 'y': 2},
    {'x': 1,  'y': 1},
    {'x': 1,  'y': -3}
]

# Lista para resultados
results = []

# Cálculo de la red para cada par (x, y)
for sample in data:
    X_VAL = sample['x']
    Y_VAL = sample['y']
    # Cálculo de las neuronas ocultas (sin activación adicional)
    h = {
        'h0': W['h0']['x'] * X_VAL + W['h0']['y'] * Y_VAL,
        'h1': W['h1']['x'] * X_VAL + W['h1']['y'] * Y_VAL,
        'h2': W['h2']['x'] * X_VAL + W['h2']['y'] * Y_VAL
    }
    # Cálculo de la salida antes de activación
    z_linear = sum(W_out[h_i] * h[h_i] for h_i in h)
    # Aplicación de la función de salida: heaviside
    Z_OUT = heaviside(z_linear)
    # Almacenar resultado
    results.append({'x': X_VAL, 'y': Y_VAL, 'z raw': z_linear, 'z': Z_OUT})

# Crear DataFrame y mostrar resultado
df = pd.DataFrame(results)
print(df.to_string(index=False))

# Imprimir feedback por pantalla:
print("\033[34mFinalizado...\033[0m \033[33mcalculo_red_neuronal\033[0m")

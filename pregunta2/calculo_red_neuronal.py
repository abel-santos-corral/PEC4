import pandas as pd

# Definición de la función heaviside

def heaviside(u):
    return 1 if u > 0 else 0

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
    x_val = sample['x']
    y_val = sample['y']
    # Cálculo de las neuronas ocultas (sin activación adicional)
    h = {
        'h0': W['h0']['x'] * x_val + W['h0']['y'] * y_val,
        'h1': W['h1']['x'] * x_val + W['h1']['y'] * y_val,
        'h2': W['h2']['x'] * x_val + W['h2']['y'] * y_val
    }
    # Cálculo de la salida antes de activación
    z_linear = sum(W_out[h_i] * h[h_i] for h_i in h)
    # Aplicación de la función de salida: heaviside
    z_out = heaviside(z_linear)
    # Almacenar resultado
    results.append({'x': x_val, 'y': y_val, 'z': z_out})

# Crear DataFrame y mostrar resultado
df = pd.DataFrame(results)
print(df.to_string(index=False))

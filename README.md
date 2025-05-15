# PEC4

Repositorio de Github para hacer la PEC4 de la asignatura de Inteligencia artificial

# Table of Contents
1. [Configuración de VS Code](#configuración-de-vs-code)
2. [Ejecutar cálculo red neuronal](#ejecutar-calculo-red-neuronal)
3. [Ejecutar análisis de k-means](#ejecutar-analisis-de-k-means)

# Configuración de VS Code

Para configurar VS Code y tener el entorno listo, sigue estos pasos.

## Crear el entorno virtual

Primero, ve a la carpeta del proyecto y ejecuta:

``` 
python -m venv venv
```

## Activar el entorno virtual

Depende del sistema operativo (OS).

__Linux__

```
source venv/bin/activate
```

__Windows (Power shell)__

```
venv\Scripts\Activate.ps1
```

__Windows (Command prompt)__

```
venv\Scripts\activate
```

## Instalar dependencias

En este caso no es necesario, pero lo dejamos comentado para reutilizarlo en otros proyectos:

```
pip install -r requirements.txt
```

# Ejecutar cálculo red neuronal

Para ejecutar el programa de cálculo de red neuronal se puede hacer con:

```
python3 pregunta2/calculo_red_neuronal.py
```

# Ejecutar análisis de k-means

Para ejecutar el programa de análisis de k-means se puede hacer con:

```
python3 pregunta3/analize_kmeans.py
```

El programa va a generar una salida por pantalla. Se tiene en cuenta lo siguiente:

* Tamaño de muestra: 10, 50 y 150
* Valor de k: 1, 2 y 5

Además, como cada ejecución se generan esos datos cogiendo al azar del repositorio Iris, los resultados pueden ser diferentes.

Se generan imágenes para pétalo y sépalo (se han hecho 2D para proveer una buena comprensión de los datos). 

Las imágenes se guardan en el folder pregunta3/imagenes/kmeans

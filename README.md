# PEC4

Repositorio de Github para hacer la PEC4 de la asignatura de Inteligencia artificial

# Table of Contents
1. [Configuración de VS Code](#configuracion-de-vs-code)

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

# Ejecutar cálculo red neuronal

Para ejecutar el programa de cálculo de red neuronal se puede hacer con:

```
python3 pregunta2/calculo_red_neuronal.py
```

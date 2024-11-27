## Paso 1: Limpieza de datos

### Ejecución
Ejecuta el siguiente script para realizar la limpieza de los datos:

```bash
python scripts/limpieza_datos.py
```

## Resultado
El script generará un archivo limpio llamado datos_limpios.csv que se guardará en la carpeta data/.

## Paso 2: Entrenamiento del modelo

### Ejecución
Ejecuta el siguiente script para entrenar el modelo de recomendación:

```bash
python scripts/entrenamiento_modelo.py
```
## Resultado

El script entrenará un modelo basado en embeddings y lo guardará como un archivo llamado modelo_recomendacion.h

## Paso 3: Evaluación del modelo

### Ejecución
Ejecuta el siguiente script para evaluar el modelo de recomendación:

```bash
python scripts/evaluacion_modelo.py
```

## Resultado
El script calculará la métrica HIT RATIO @20 y mostrará el resultado en la consola.
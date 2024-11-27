import pandas as pd
import numpy as np
import tensorflow as tf
from sklearn.model_selection import train_test_split

def cargar_datos(ruta):
    """Carga los datos limpios."""
    return pd.read_csv(ruta)

def cargar_modelo(ruta):
    """Carga el modelo entrenado."""
    return tf.keras.models.load_model(ruta)

def evaluar_hit_ratio(modelo, X_test, y_test, top_k=20):
    """
    Calcula el HIT RATIO @K.

    Args:
        modelo (tf.keras.Model): Modelo entrenado.
        X_test (array): Datos de prueba (usuarios y libros).
        y_test (array): Valores reales.
        top_k (int): Número de predicciones a considerar.

    Returns:
        hit_ratio: Porcentaje de aciertos en las predicciones.
    """
    hits = 0
    total = len(X_test)
    
    # Realizar predicciones
    predicciones = modelo.predict([X_test[:, 0], X_test[:, 1]])
    
    # Obtener los índices ordenados por probabilidad (de mayor a menor)
    indices_ordenados = np.argsort(-predicciones.flatten())[:top_k]
    
    # Comparar con las etiquetas reales
    for i in indices_ordenados:
        if y_test[i] == 1:
            hits += 1
    
    hit_ratio = hits / total
    return hit_ratio

if __name__ == "__main__":
    # Cargar los datos limpios
    ruta_datos = r"Proyecto_Motor_Recomendacion\data\datos_limpios.csv"
    datos = cargar_datos(ruta_datos)
    
    # Cargar el modelo entrenado
    ruta_modelo = r"Proyecto_Motor_Recomendacion\models\modelo_recomendacion.h5"
    modelo = cargar_modelo(ruta_modelo)
    
    if 'Rating' not in datos.columns:
        datos['Rating'] = 1  # Asignar valor predeterminado para indicar interacciones



    # Preparar los datos de entrada y salida
    X = datos[['Usuario_ID', 'Libro_ID']].values
    y = datos['Rating'].values  

    # Dividir los datos en entrenamiento y prueba
    _, X_test, _, y_test = train_test_split(X, y, test_size=0.2, random_state=42)
    
    # Evaluar el modelo con la métrica HIT RATIO @20
    hit_ratio = evaluar_hit_ratio(modelo, X_test, y_test, top_k=20)
    
    # Mostrar el resultado
    print(f"HIT RATIO @20: {hit_ratio:.2f}")

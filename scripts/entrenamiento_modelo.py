import pandas as pd
import tensorflow as tf
from keras._tf_keras.keras.models import Model
from keras._tf_keras.keras.layers import Input, Embedding, Flatten, Dot, Dense
from sklearn.model_selection import train_test_split



def cargar_datos(ruta):
    """Carga los datos limpios."""
    return pd.read_csv(ruta)

def crear_modelo(num_usuarios, num_libros, embedding_dim=50):
    """
    Crea un modelo de recomendación basado en embeddings.
    
    Args:
        num_usuarios (int): Número de usuarios únicos.
        num_libros (int): Número de libros únicos.
        embedding_dim (int): Dimensión de los embeddings.

    Returns:
        modelo: Modelo de recomendación compilado.
    """
    # Entradas del modelo
    usuario_input = Input(shape=(1,), name='usuario_input')
    libro_input = Input(shape=(1,), name='libro_input')
    
    # Embeddings
    usuario_embedding = Embedding(input_dim=num_usuarios, output_dim=embedding_dim, name='usuario_embedding')(usuario_input)
    libro_embedding = Embedding(input_dim=num_libros, output_dim=embedding_dim, name='libro_embedding')(libro_input)
    
    # Aplanar los vectores
    usuario_vec = Flatten()(usuario_embedding)
    libro_vec = Flatten()(libro_embedding)
    
    # Producto punto para obtener la similitud
    dot_product = Dot(axes=1)([usuario_vec, libro_vec])
    
    # Salida del modelo (escala de 0 a 1)
    salida = Dense(1, activation='sigmoid')(dot_product)
    
    # Crear el modelo
    modelo = Model(inputs=[usuario_input, libro_input], outputs=salida)
    modelo.compile(optimizer='adam', loss='binary_crossentropy', metrics=['accuracy'])
    
    return modelo

def entrenar_modelo(df, modelo, epochs=10, batch_size=32):
    """
    Entrena el modelo de recomendación.

    Args:
        df (DataFrame): DataFrame con los datos codificados (Usuario_ID, Libro_ID, Rating).
        modelo (Model): Modelo a entrenar.
        epochs (int): Número de épocas.
        batch_size (int): Tamaño del lote.

    Returns:
        historia: Objeto con el historial de entrenamiento.
    """
    # Crear una columna 'Rating' si no existe
    if 'Rating' not in df.columns:
        df['Rating'] = 1  
    
    # Preparar los datos de entrada y salida
    X = df[['Usuario_ID', 'Libro_ID']].values
    y = df['Rating'].values  # Columna de interacciones
    
    # Dividir en entrenamiento y prueba
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)
    
    # Entrenar el modelo
    historia = modelo.fit(
        [X_train[:, 0], X_train[:, 1]], y_train,
        validation_data=([X_test[:, 0], X_test[:, 1]], y_test),
        epochs=epochs,
        batch_size=batch_size
    )
    
    return historia

if __name__ == "__main__":
    # Cargar los datos limpios
    ruta_datos = r"Proyecto_Motor_Recomendacion\data\datos_limpios.csv"
    datos = cargar_datos(ruta_datos)
    
    # Verificar el número de usuarios y libros únicos
    num_usuarios = datos['Usuario_ID'].nunique()
    num_libros = datos['Libro_ID'].nunique()
    
    # Crear el modelo
    modelo = crear_modelo(num_usuarios, num_libros)
    
    # Entrenar el modelo
    historia = entrenar_modelo(datos, modelo, epochs=10, batch_size=32)
    
    # Guardar el modelo entrenado
    modelo.save(r"Proyecto_Motor_Recomendacion\models\modelo_recomendacion.h5")
    print("Proyecto_Motor_Recomendacion\models\modelo_recomendacion.h5'")

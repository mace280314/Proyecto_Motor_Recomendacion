import pandas as pd
from sklearn.preprocessing import LabelEncoder

def cargar_datos(ruta_archivo):
    """Carga los datos desde el archivo Excel."""
    return pd.read_excel(ruta_archivo)

def limpiar_datos(df):
    """Limpia y prepara los datos."""
    # Filtrar libros leídos una sola vez
    libros_frecuencia = df['Título'].value_counts()
    libros_validos = libros_frecuencia[libros_frecuencia > 1].index
    df = df[df['Título'].isin(libros_validos)]
    
    # Filtrar usuarios que hayan leído más de un libro
    usuarios_frecuencia = df['Identificador Socio'].value_counts()
    usuarios_validos = usuarios_frecuencia[usuarios_frecuencia > 1].index
    df = df[df['Identificador Socio'].isin(usuarios_validos)]
    
    # Eliminar libros duplicados por usuario
    df = df.drop_duplicates(subset=['Identificador Socio', 'Título'])
    
    return df

def codificar_datos(df):
    """Codifica usuarios y libros en números."""
    le_usuario = LabelEncoder()
    le_libro = LabelEncoder()
    
    df['Usuario_ID'] = le_usuario.fit_transform(df['Identificador Socio'])
    df['Libro_ID'] = le_libro.fit_transform(df['Título'])
    
    return df, le_usuario, le_libro

def guardar_datos(df, ruta_salida):
    """Guarda el dataset limpio."""
    df.to_csv(ruta_salida, index=False)

if __name__ == "__main__":
    ruta_entrada = r"Proyecto_Motor_Recomendacion\data\CIRCULA_23_TRANSF_S.xlsx"
    ruta_salida = r"Proyecto_Motor_Recomendacion\data\datos_limpios.csv"
    

    datos = pd.read_excel(ruta_entrada, sheet_name="CIRCULA_2023")
    print("Columnas disponibles:", datos.columns)
    datos_limpios = limpiar_datos(datos)
    datos_codificados, _, _ = codificar_datos(datos_limpios)
    guardar_datos(datos_codificados, ruta_salida)

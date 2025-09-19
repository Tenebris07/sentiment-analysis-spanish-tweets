import pandas as pd
import re
from sklearn.model_selection import train_test_split

# Cargar dataset
def cargar_datos(ruta='data/sentiment_analysis_dataset.csv'):
    df = pd.read_csv(ruta)
    print("Primeras filas del dataset:")
    print(df.head())
    print("\nDistribución de etiquetas:")
    print(df['sentiment'].value_counts())
    return df

# Función para limpiar texto
def limpiar_texto(texto):
    texto = texto.lower()
    texto = re.sub(r'http\S+', '', texto)  # eliminar URLs
    texto = re.sub(r'[^a-záéíóúüñ\s]', '', texto)  # quitar caracteres no alfabéticos
    texto = re.sub(r'\s+', ' ', texto).strip()  # quitar espacios extras
    return texto

# Preprocesar dataset (limpiar texto)
def preprocesar_df(df):
    df['text_clean'] = df['text'].apply(limpiar_texto)
    return df

# Dividir datos en entrenamiento y prueba
def dividir_datos(df, test_size=0.2, random_state=42):
    X_train, X_test, y_train, y_test = train_test_split(
        df['text_clean'], df['sentiment'], test_size=test_size, random_state=random_state
    )
    return X_train, X_test, y_train, y_test

if __name__ == "__main__":
    df = cargar_datos()
    df = preprocesar_df(df)
    X_train, X_test, y_train, y_test = dividir_datos(df)
    print(f"\nNúmero de ejemplos en entrenamiento: {len(X_train)}")
    print(f"Número de ejemplos en prueba: {len(X_test)}")

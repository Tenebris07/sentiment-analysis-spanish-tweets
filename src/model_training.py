import pandas as pd
import pickle
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import classification_report, accuracy_score
from sklearn.model_selection import train_test_split
from data_preprocessing import cargar_datos, preprocesar_df, dividir_datos

def entrenar_modelo():
    # Cargar y preprocesar datos
    df = cargar_datos()
    df = preprocesar_df(df)
    
    # Dividir datos
    X_train, X_test, y_train, y_test = dividir_datos(df)

    # Transformar texto a vectores TF-IDF
    vectorizer = TfidfVectorizer(max_features=5000)
    X_train_vect = vectorizer.fit_transform(X_train)
    X_test_vect = vectorizer.transform(X_test)

    # Entrenar modelo de regresión logística
    modelo = LogisticRegression(max_iter=1000)
    modelo.fit(X_train_vect, y_train)

    # Predecir y evaluar
    y_pred = modelo.predict(X_test_vect)

    print("Reporte de clasificación:\n", classification_report(y_test, y_pred))
    print(f"Exactitud: {accuracy_score(y_test, y_pred):.4f}")

    return modelo, vectorizer

if __name__ == "__main__":
    entrenar_modelo()

if __name__ == "__main__":
    modelo, vectorizador = entrenar_modelo()

    # Guardar modelo y vectorizador
    with open('modelo.pkl', 'wb') as f_model:
        pickle.dump(modelo, f_model)

    with open('vectorizador.pkl', 'wb') as f_vec:
        pickle.dump(vectorizador, f_vec)

    print("Modelo y vectorizador guardados correctamente.")
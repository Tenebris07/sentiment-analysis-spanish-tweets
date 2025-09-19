import sys
import pickle
from data_preprocessing import limpiar_texto

def cargar_modelo_y_vectorizador(model_path='modelo.pkl', vector_path='vectorizador.pkl'):
    with open(model_path, 'rb') as f:
        modelo = pickle.load(f)
    with open(vector_path, 'rb') as f:
        vectorizador = pickle.load(f)
    return modelo, vectorizador

def predecir_sentimiento(texto, modelo, vectorizador):
    texto_limpio = limpiar_texto(texto)
    vector = vectorizador.transform([texto_limpio])
    prediccion = modelo.predict(vector)[0]
    return prediccion

if __name__ == "__main__":
    if len(sys.argv) < 2:
        print("Por favor, escribe un texto para clasificar:")
        texto = input()
    else:
        texto = sys.argv[1]
    
    modelo, vectorizador = cargar_modelo_y_vectorizador()
    resultado = predecir_sentimiento(texto, modelo, vectorizador)
    print(f"Texto: {texto}")
    print(f"Sentimiento predicho: {resultado}")

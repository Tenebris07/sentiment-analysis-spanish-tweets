import os

def descargar_dataset():
    # Comando para descargar y descomprimir dataset de Kaggle
    comando = "kaggle datasets download philipsanm/sentiment-analysis-in-spanish-tweets -p data/ --unzip"
    os.system(comando)
    print("Descarga completa. Dataset ubicado en carpeta 'data/'.")

if __name__ == "__main__":
    descargar_dataset()

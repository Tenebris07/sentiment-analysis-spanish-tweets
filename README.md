# Análisis de Sentimientos en Español con Python

Este proyecto implementa un clasificador de sentimientos basado en aprendizaje automático para textos en español, especialmente tweets. Permite clasificar el sentimiento general en varias categorías emocionales.

## Contenido del proyecto

- `data/`: Contiene el dataset (tweets etiquetados con sentimiento).
- `notebooks/`: Análisis exploratorio y experimentación inicial.
- `src/`: Código fuente para descargar datos, preprocesar texto, entrenar modelo y hacer predicciones.
- `requirements.txt`: Dependencias de Python necesarias.
- `README.md`: Este archivo.

## Instalación y configuración

1. Clona el repositorio
2. Crea y activa un entorno virtual (recomendado)
3. Instala las dependencias
pip install -r requirements.txt
4. Configura la API de Kaggle para descargar el dataset (desde tu cuenta Kaggle > Perfil > API).
5. Descarga el dataset:
python src/download_data.py

## Uso del proyecto
1. Preprocesa y explora los datos:
python src/data_preprocessing.py

2. Entrena el modelo y guárdalo:
python src/model_training.py

3. Realiza predicciones con textos nuevos:
ejemplo:
python src/predict.py "Esto es asombroso y me encantó"



## Resultados y evaluación

El modelo actual es multiclase con categorías como: serene, mad, powerful, sad, joyful, scared. La exactitud aproximada alcanza 62%, con bastante equilibrio entre precisión y recall para cada clase.

## Próximos pasos

- Mejorar la limpieza y feature extraction.
- Probar otros algoritmos (SVM, Random Forest, Deep Learning).
- Crear una interfaz web sencilla para hacer demo en vivo (por ejemplo con Streamlit).
- Explorar modelos preentrenados de lenguaje natural para mejorar la precisión.

## Licencia

MIT License

---

Proyecto desarrollado para portafolio personal de análisis de datos e inteligencia artificial.









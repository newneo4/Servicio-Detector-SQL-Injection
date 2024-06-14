from flask import Flask, request, jsonify
import pickle
from sklearn.feature_extraction.text import CountVectorizer
from tensorflow.keras.models import load_model
import numpy as np
from flasgger import Swagger

app = Flask(__name__)
swagger = Swagger(app)

# Cargar el vectorizador guardado
def load_vectorizer():
    with open('vectorizer.pkl', 'rb') as f:
        return pickle.load(f)

vectorizer = load_vectorizer()

# Cargar el modelo entrenado (suponiendo que sea un modelo de TensorFlow Keras)
def load_trained_model():
    return load_model('sql_injection_model2.h5')

model = load_trained_model()

@app.route('/')
def index():
    """
    Bienvenido a la API de predicción de SQL Injection.
    ---
    responses:
      200:
        description: Página de inicio de la API
    """
    return 'Bienvenido a la API de predicción de SQL Injection. Utilice el endpoint /prediccion para obtener predicciones.'

@app.route('/prediccion', methods=['POST'])
def prediccion():
    """
    Endpoint para obtener una predicción de SQL Injection.
    ---
    parameters:
      - name: body
        in: body
        required: true
        schema:
          type: object
          required:
            - query
          properties:
            query:
              type: string
              description: La consulta SQL para analizar
              example: "SELECT * FROM users WHERE id = 1"
    responses:
      200:
        description: Resultado de la predicción
        schema:
          type: object
          properties:
            prediccion:
              type: integer
              description: La predicción, 0 o 1
              example: 1
      400:
        description: Error en la solicitud
        schema:
          type: object
          properties:
            error:
              type: string
              description: Mensaje de error
              example: "No se encontró la consulta en los datos JSON"
      500:
        description: Error interno del servidor
        schema:
          type: object
          properties:
            error:
              type: string
              description: Mensaje de error
              example: "Error al realizar la predicción"
    """
    data = request.get_json()

    if 'query' not in data:
        return jsonify({"error": "No se encontró la consulta en los datos JSON"}), 400

    query = data['query']

    try:
        query_transformed = vectorizer.transform([query]).toarray()
    except ValueError:
        return jsonify({"error": "La consulta no se pudo transformar correctamente. Asegúrate de que el vectorizador esté ajustado correctamente."}), 400

    try:
        query_transformed = query_transformed.reshape(-1, 1, query_transformed.shape[1])

        prediction = model.predict(query_transformed)

        # Redondear y convertir la predicción a entero (0 o 1)
        prediction_label = int(round(prediction[0][0]))

        # Preparar la respuesta JSON
        response = {"prediccion": prediction_label} 
        
        return jsonify(response), 200
    
    except ValueError as ve:
        return jsonify({"error": f"Error al realizar la predicción: {ve}"}), 500

if __name__ == '__main__':
    app.run(debug=True)

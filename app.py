from flask import Flask, request, jsonify
import pickle
from sklearn.feature_extraction.text import CountVectorizer
from tensorflow.keras.models import load_model
import numpy as np
from flask_swagger_ui import get_swaggerui_blueprint


app = Flask(__name__)

SWAGGER_URL = '/api/docs'
API_URL = '/static/swagger.json'

swaggerui_blueprint = get_swaggerui_blueprint(
    SWAGGER_URL, 
    API_URL,
    config={  
        'app_name': "Test application"
    },
)

app.register_blueprint(swaggerui_blueprint)

def load_vectorizer():
    with open('vectorizer.pkl', 'rb') as f:
        return pickle.load(f)

vectorizer = load_vectorizer()

def load_trained_model():
    return load_model('sql_injection_model2.h5')

model = load_trained_model()

@app.route('/prediccion', methods=['POST'])
def prediccion():
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

        prediction_label = int(round(prediction[0][0]))

        response = {"prediccion": prediction_label} 
        
        return jsonify(response), 200
    
    except ValueError as ve:
        return jsonify({"error": f"Error al realizar la predicción: {ve}"}), 500

@app.route('/testing', methods=['POST'])
def testing():
    data = request.get_json()

    if not data or 'query' not in data:
        return jsonify({"error": "No se encontró 'query' en los datos JSON"}), 400

    query = data['query']

    try:
        query_value = int(query)
    except ValueError:
        return jsonify({"error": "'query' debe ser un número entero"}), 400

    response = {"respuesta": query_value * 2}
    return jsonify(response), 200


if __name__ == '__main__':
    app.run(debug=True)

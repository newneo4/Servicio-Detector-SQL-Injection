from flask import Flask, request, jsonify
import pickle
from sklearn.feature_extraction.text import CountVectorizer
from tensorflow.keras.models import load_model
import numpy as np

app = Flask(__name__)

# Cargar el vectorizador y el modelo
def load_vectorizer():
    with open('vectorizer.pkl', 'rb') as f:
        return pickle.load(f)

vectorizer = load_vectorizer()

def load_trained_model():
    return load_model('sql_injection_model2.h5')

model = load_trained_model()

# Ruta para la predicción de SQL injection
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

if __name__ == '__main__':
    app.run(debug=True)

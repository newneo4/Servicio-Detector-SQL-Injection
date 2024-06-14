from flask import Flask, request, jsonify
import pickle
from sklearn.feature_extraction.text import CountVectorizer
from tensorflow.keras.models import load_model
import numpy as np

app = Flask(__name__)

# Cargar el vectorizador guardado
try:
    with open('vectorizer.pkl', 'rb') as f:
        vectorizer = pickle.load(f)
except FileNotFoundError:
    print("El archivo vectorizer.pkl no se encontró. Asegúrate de haber guardado el vectorizador previamente.")
    exit(1)

# Cargar el modelo entrenado (suponiendo que sea un modelo de TensorFlow Keras)
try:
    model = load_model('sql_injection_model2.h5')  
except FileNotFoundError:
    print("El archivo del modelo no se encontró. Asegúrate de haber guardado el modelo previamente.")
    exit(1)

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

        # Redondear y convertir la predicción a entero (0 o 1)
        prediction_label = int(round(prediction[0][0]))

        # Preparar la respuesta JSON
        response = {"prediccion": prediction_label} 
        
        return jsonify(response), 200
    
    except ValueError as ve:
        return jsonify({"error": f"Error al realizar la predicción: {ve}"}), 500

if __name__ == '__main__':
    app.run(debug=True)

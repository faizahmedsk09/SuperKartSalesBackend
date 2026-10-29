
# Import data manipulation libraries
import numpy as np
import pandas as pd

# For serialization
import joblib

# Flask API
from flask import Flask, request, jsonify

# Import logging
import sys
import logging

# Initialize the Flask app with a name
superkart_api = Flask("superkart_sales_app")

# Debug info
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s',
    handlers=[
        logging.StreamHandler(sys.stdout)
    ]
)
logger = logging.getLogger(__name__)

logger.info(f"Module name: {__name__}")
logger.info(f"Flask app name: {superkart_api.name}")
logger.info(f"Root path: {superkart_api.root_path}")

# Load the finalized trained model pipeline
import os
model_path = "superkart_revenue_predictor_model_v1_0.joblib"
if not os.path.exists(model_path):
    model_path = "forecast_superkart_sales_model.joblib"

logger.info(f"Loading model from: {model_path}")
model = joblib.load(model_path)

# Define the exact list of features expected by the model pipeline
MODEL_FEATURES = [
    'Product_Weight',
    'Product_Sugar_Content',
    'Product_Allocated_Area',
    'Product_MRP',
    'Store_Size',
    'Store_Location_City_Type',
    'Store_Type',
    'Store_Age_Years',
    'Product_Type_Category',
    'Product_Id_char'
]

# Define a route for the home page
@superkart_api.route('/', methods=['GET'])
def home():
    logger.info("Home endpoint accessed")
    html = """
      <!DOCTYPE html>
      <html>
      <head>
        <title>SuperKart Sales API</title>
        <style>
          body {
            font-family: Arial, sans-serif;
            text-align: center;
            padding: 50px;
            background-color: #f4f4f4;
          }
          h1 {
            color: #333;
            font-size: 3em;
          }
          p {
            color: #666;
            font-size: 1.5em;
            margin-top: 20px;
          }
        </style>
      </head>
      <body>
        <h1>Welcome to SuperKart Sales Prediction API.</h1>
        <p>To obtain sales predictions, please send a POST request to `/v1/predict`.</p>
      </body>
      </html>
    """
    return html

# Define an endpoint to predict a product sale
@superkart_api.route('/v1/predict', methods=['POST'])
def predict_sales():
    try:
      # Get the JSON data from the request body
      data = request.get_json()

      # Extract relevant product features from the input data.
      sample = {
        'Product_Weight': data['Product_Weight'],
        'Product_Sugar_Content': data['Product_Sugar_Content'],
        'Product_Allocated_Area': data['Product_Allocated_Area'],
        'Product_MRP': data['Product_MRP'],
        'Store_Size': data['Store_Size'],
        'Store_Location_City_Type': data['Store_Location_City_Type'],
        'Store_Type': data['Store_Type'],
        'Store_Age_Years': data['Store_Age_Years'],
        'Product_Type_Category': data['Product_Type_Category'],
        'Product_Id_char': data['Product_Id_char']
      }

      # Convert the extracted data into a DataFrame
      input_data = pd.DataFrame([sample])
      logger.debug(f"Input DataFrame:\n{input_data}")

      # Make a prediction using the trained model
      prediction = model.predict(input_data).tolist()[0]

      # Return the prediction as a JSON response
      return jsonify({'Sales': prediction})

    except KeyError as e:
      return jsonify({'error': f'Missing key: {str(e)}'}), 400
    except Exception as e:
      return jsonify({'error': f'Prediction failed: {str(e)}' }), 500

# Define an endpoint for batch prediction (POST request)
@superkart_api.route('/v1/predictbatch', methods=['POST'])
def predict_sales_batch():
    # Get the uploaded CSV file from the request
    file = request.files.get('file')
    if file is None:
        return jsonify({'error': 'Missing file upload: expected a CSV file in the "file" field'}), 400

    # Read the CSV file into a Pandas DataFrame
    input_data = pd.read_csv(file)

    # Keep an optional row identifier out of the model's feature columns.
    product_ids = input_data['id'].tolist() if 'id' in input_data else list(range(len(input_data)))
    model_input = input_data.drop(columns=['id'], errors='ignore')

    # Enforce correct column order to align with preprocessing pipeline
    model_input = model_input[MODEL_FEATURES]

    # Make predictions for all products in the DataFrame
    predicted_sales = model.predict(model_input).tolist()
    output_dict = dict(zip(product_ids, predicted_sales))

    # Return the predictions dictionary as a JSON response
    return jsonify(output_dict)

# Run the Flask app
if __name__ == '__main__':
    superkart_api.run(debug=True)

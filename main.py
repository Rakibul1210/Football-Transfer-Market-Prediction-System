from flask import Flask, request, jsonify
import joblib
import pandas as pd
from sklearn.preprocessing import OneHotEncoder
from sklearn.compose import ColumnTransformer
from flask_cors import CORS

app = Flask(__name__)
CORS(app)

# Load the pre-trained model and preprocessing objects when the server starts
model = joblib.load('model/trained_rf_model_with_categorical.joblib')
preprocessor = joblib.load('model/preprocessor_with_categorical.joblib')

# Define your API endpoint
@app.route('/predict_transfer_fee', methods=['POST'])
def predict_transfer_fee():

    print("We are hereee")
    try:
        # Check if the request contains JSON data
        if request.is_json:
            # Receive player stats dynamically from the API request
            player_data = request.get_json().get('playerData')

            if player_data:
                print("Received Player Stats:")
                print(player_data)

                # Create a DataFrame from the dictionary
                player_df = pd.DataFrame([player_data])

                # Extract the categorical columns for preprocessing
                categorical_cols = ['Nation', 'Pos', 'Squad', 'Comp']

                # Use the preprocessor to transform the input data
                player_df_encoded = preprocessor.transform(player_df)

                # Make predictions using the loaded model
                predicted_transfer_fee = model.predict(player_df_encoded)

                print("Predicted Transfer Fee:", predicted_transfer_fee[0])
                print("We are going back")
                return jsonify({
                    "name": player_data.get("Name"),
                    "transfer_fee": float(predicted_transfer_fee[0])
                })
            else:
                return jsonify({"error": "Invalid playerData structure"})

        else:
            return jsonify({"error": "Invalid request format. Must be JSON."})

    except Exception as e:
        return jsonify({"error": str(e)})

if __name__ == '__main__':
    app.run(host='0.0.0.0')

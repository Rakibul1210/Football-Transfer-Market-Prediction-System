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
    try:
        # # Check if the request contains JSON data
        # if request.is_json:
        #     # Receive player stats dynamically from the API request
        #     player_stats = request.get_json()
        # else:
        #     # Use hardcoded player stats for testing

        player_stats = {
            'Name': 'Antony',
            'Market_value': 35,
            'Transfer_fee': 95,
            'Rk': 105,
            'Nation': 'BRA',
            'Pos': 'FW',
            'Squad': 'Manchester Utd',
            'Comp': 'Premier League',
            'Age': 22,
            'MP': 12,
            'Starts': 11,
            'Min': 908,
            'Goals': 3,
            'SoT%': 42.4,
            'G/Sh': 0.09,
            'ShoDist': 20.4,
            'ShoFK': 0,
            'ShoPK': 0,
            'PKatt': 0,
            'PasTotCmp%': 77.2,
            'Assists': 0,
            'PasAss': 0.79,
            'PasProg': 2.77,
            'PasAtt': 40.9,
            'PasLive': 40.4,
            'PasDead': 0.5,
            'PasCrs': 1.78,
            'CK': 0.1,
            'PasCmp': 31.6,
            'GCA': 0.1,
            'TklWon': 0.3,
            'TklDef3rd': 0.69,
            'Int': 0.4,
            'Clr': 0.3,
            'Err': 0,
            'Touches': 52.6,
            'Carries': 37.8,
            'Car3rd': 3.47,
            'Rec': 40,
            'RecProg': 7.92,
            'CrdY': 0.3,
            'CrdR': 0,
            'Fls': 0.4,
            'Crs': 1.78,
            'TklW': 0.3,
            'PKwon': 0,
            'PKcon': 0,
            'OG': 0,
            'AerWon%': 27.3,
        }

        player_stats = {
        'Name': 'Erling Haaland',
        'Market_value': 150,
        'Transfer_fee': 60,
        'Rk': 1058,
        'Nation': 'NOR',
        'Pos': 'FW',
        'Squad': 'Manchester City',
        'Comp': 'Premier League',
        'Age': 22,
        'MP': 20,
        'Starts': 19,
        'Min': 1636,
        'Goals': 25,
        'SoT%': 51.4,
        'G/Sh': 0.3,
        'ShoDist': 12.6,
        'ShoFK': 0,
        'ShoPK': 0.22,
        'PKatt': 0.22,
        'PasTotCmp%': 73.7,
        'Assists': 0.16,
        'PasAss': 0.93,
        'PasProg': 1.87,
        'PasAtt': 15.9,
        'PasLive': 13.6,
        'PasDead': 2.09,
        'PasCrs': 0.27,
        'CK': 0,
        'PasCmp': 11.7,
        'GCA': 0.38,
        'TklWon': 0.05,
        'TklDef3rd': 0,
        'Int': 0.11,
        'Clr': 0.55,
        'Err': 0,
        'Touches': 24.2,
        'Carries': 12.6,
        'Car3rd': 0.27,
        'Rec': 18.1,
        'RecProg': 5.05,
        'CrdY': 0.16,
        'CrdR': 0,
        'Fls': 0.82,
        'Crs': 0.27,
        'TklW': 0.05,
        'PKwon': 0.05,
        'PKcon': 0,
        'OG': 0,
        'AerWon%': 52.1,
        }

        print("Received Player Stats:")
        print(player_stats)

        # Create a DataFrame from the dictionary
        player_df = pd.DataFrame([player_stats])

        # Extract the categorical columns for preprocessing
        categorical_cols = ['Nation', 'Pos', 'Squad', 'Comp']

        # Use the preprocessor to transform the input data
        player_df_encoded = preprocessor.transform(player_df)

        # Make predictions using the loaded model
        predicted_transfer_fee = model.predict(player_df_encoded)

        print("Predicted Transfer Fee:", predicted_transfer_fee[0])

        return jsonify({
            "name": player_stats["Name"],
            "transfer_fee": float(predicted_transfer_fee[0])
        })

    except Exception as e:
        return jsonify({"error": str(e)})

if __name__ == '__main__':
    app.run(debug=True)

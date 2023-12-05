import joblib
import pandas as pd

# Load the pre-trained model
model = joblib.load('model/trained_rf_model.joblib')

player_stats = {
    'Name': 'Erling Haaland',  # No need for a list since we're predicting one player
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

# Create a DataFrame from the dictionary
player_df = pd.DataFrame([player_stats])
player_df = player_df.drop([ 'Nation', 'Pos', 'Squad', 'Comp'], axis=1)   # For Now
player_df = player_df.drop(['Market_value', 'Name','Transfer_fee','Rk'], axis=1) # Permanent
print(player_df)

# Make predictions using the loaded model
predicted_transfer_fee = model.predict(player_df)
print("Pedicted market prize for", player_stats["Name"][0], " :", predicted_transfer_fee[0], " Milion")
print("-----END-----")
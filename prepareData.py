import pandas as pd

player_stat = 'data/player-stats-22-23.csv'
transfer_list = "data/player-transfer-22-23.csv"

print("Hello World")

df_transfer = pd.read_csv(transfer_list)
df_stats = pd.read_csv(player_stat)

# print(df_transfer['Name'])
# print(df_stats['Name'])


# Replacing 'free transfer' & 'loan transfer' with value 0
df_transfer['Transfer_fee'] = df_transfer['Transfer_fee'].replace('free transfer', 0)
df_transfer['Transfer_fee'] = df_transfer['Transfer_fee'].replace('loan transfer', 0)
df_transfer['Transfer_fee'] = df_transfer['Transfer_fee'].replace('?', 0)


merged_df = df_transfer.merge(df_stats, on='Name')
print(merged_df)
# df = df.drop(['Name', 'Nation', 'Pos', 'Squad', 'Comp', 'Market_value','Rk'], axis=1)
# print(merged_df)

output_file = 'data/final-data-22-23.csv'

merged_df.to_csv(output_file, index=False)
import pandas as pd
import numpy as np

# Load preprocessed data
df = pd.read_csv("jld_aqi_filled.csv")

# CPCB breakpoints for AQI calculation
breakpoints = {
    'PM2.5': [
        (0, 30, 0, 50), (31, 60, 51, 100), (61, 90, 101, 200),
        (91, 120, 201, 300), (121, 250, 301, 400), (251, 350, 401, 500)
    ],
    'PM10': [
        (0, 50, 0, 50), (51, 100, 51, 100), (101, 250, 101, 200),
        (251, 350, 201, 300), (351, 430, 301, 400), (431, 500, 401, 500)
    ],
    'NO2': [
        (0, 40, 0, 50), (41, 80, 51, 100), (81, 180, 101, 200),
        (181, 280, 201, 300), (281, 400, 301, 400), (401, 500, 401, 500)
    ],
    'SO2': [
        (0, 40, 0, 50), (41, 80, 51, 100), (81, 380, 101, 200),
        (381, 800, 201, 300), (801, 1600, 301, 400), (1601, 2000, 401, 500)
    ],
    'CO': [
        (0.0, 1.0, 0, 50), (1.1, 2.0, 51, 100), (2.1, 10.0, 101, 200),
        (10.1, 17.0, 201, 300), (17.1, 34.0, 301, 400), (34.1, 50.0, 401, 500)
    ],
    'Ozone': [
        (0, 50, 0, 50), (51, 100, 51, 100), (101, 168, 101, 200),
        (169, 208, 201, 300), (209, 748, 301, 400), (749, 1000, 401, 500)
    ]
}

# Sub-index calculation function
def calculate_sub_index(value, bps):
    for bp_low, bp_high, aqi_low, aqi_high in bps:
        if bp_low <= value <= bp_high:
            return ((aqi_high - aqi_low) / (bp_high - bp_low)) * (value - bp_low) + aqi_low
    return np.nan

# Calculate AQI sub-indices for each pollutant
df['AQI_PM2.5'] = df['PM2.5 (ug/m3)'].apply(lambda x: calculate_sub_index(x, breakpoints['PM2.5']))
df['AQI_PM10'] = df['PM10 (ug/m3)'].apply(lambda x: calculate_sub_index(x, breakpoints['PM10']))
df['AQI_NO2'] = df['NO2 (ug/m3)'].apply(lambda x: calculate_sub_index(x, breakpoints['NO2']))
df['AQI_SO2'] = df['SO2 (ug/m3)'].apply(lambda x: calculate_sub_index(x, breakpoints['SO2']))
df['AQI_CO']   = df['CO (mg/m3)'].apply(lambda x: calculate_sub_index(x, breakpoints['CO']))
df['AQI_Ozone'] = df['Ozone (ug/m3)'].apply(lambda x: calculate_sub_index(x, breakpoints['Ozone']))

# Final AQI is the maximum of all sub-indices
df['AQI'] = df[['AQI_PM2.5', 'AQI_PM10', 'AQI_NO2', 'AQI_SO2', 'AQI_CO', 'AQI_Ozone']].max(axis=1)

# Save output
df.to_csv("jld_aqi_with_aqi.csv", index=False)

print("AQI calculation complete. Output saved to 'jld_aqi_with_aqi.csv'.")



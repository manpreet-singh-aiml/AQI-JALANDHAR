import pandas as pd

# Load the CSV file
df = pd.read_csv("jld_aqi.csv")

# Convert the 'Date' column to datetime format (dd-mm-yyyy HH:MM:SS)
df['Date'] = pd.to_datetime(df['Date'], format='%d-%m-%Y %H:%M:%S', errors='coerce')

# Drop rows with invalid or missing dates
df = df.dropna(subset=['Date'])

# Sort data by date
df = df.sort_values('Date')

# Set 'Date' as index for time-series operations
df = df.set_index('Date')

# Remove duplicate timestamps
df = df[~df.index.duplicated(keep='first')]

# Create full hourly time range
full_range = pd.date_range(start=df.index.min(), end=df.index.max(), freq='h')

# Reindex to full hourly range (missing timestamps -> NaN)
df = df.reindex(full_range)

# Linear interpolation to fill missing values
# Mean/mode imputation avoided to preserve temporal dependencies
df = df.interpolate(method='linear', limit_direction='both')

# Fill any remaining edge values
df = df.ffill().bfill()

# Reset index so 'Date' becomes a column
df = df.reset_index().rename(columns={'index': 'Date'})

# Save cleaned dataset
df.to_csv("jld_aqi_filled.csv", index=False)

# Print summary
print("Preprocessing complete.")
print("Date range:", df['Date'].min(), "to", df['Date'].max())
print("Total records:", len(df))
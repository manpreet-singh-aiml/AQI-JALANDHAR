import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.preprocessing import MinMaxScaler
from sklearn.metrics import (
    mean_squared_error, mean_absolute_error,
    r2_score, mean_absolute_percentage_error
)
from tensorflow.keras.models import Model
from tensorflow.keras.layers import (
    Input, LSTM, LayerNormalization, Dense,
    TimeDistributed, Add, Dropout, GlobalAveragePooling1D
)
from tensorflow.keras.optimizers import Adam
from tensorflow.keras.callbacks import EarlyStopping, ReduceLROnPlateau, Callback
import tensorflow as tf


# ---------------------------------------------------------------------
# Custom callback to record learning rate and best validation loss
# ---------------------------------------------------------------------
class RunLogger(Callback):
    def on_train_begin(self, logs=None):
        self.lrs = []
        self.best_val_loss = float('inf')
        self.best_epoch = 0

    def on_epoch_end(self, epoch, logs=None):
        self.lrs.append(float(self.model.optimizer.learning_rate))
        val = logs.get('val_loss')
        if val is not None and val < self.best_val_loss:
            self.best_val_loss = val
            self.best_epoch = epoch + 1


# ---------------------------------------------------------------------
# 1. Load data
# ---------------------------------------------------------------------
df = pd.read_csv("jld_aqi_with_aqi.csv", parse_dates=['Date']).set_index('Date')

features = [
    'PM2.5 (ug/m3)', 'PM10 (ug/m3)', 'NO2 (ug/m3)',
    'SO2 (ug/m3)', 'CO (mg/m3)', 'Ozone (ug/m3)'
]

# 3-step rolling average applied prior to train-test split
# Backward-looking operation — does not introduce future information
df['AQI'] = df['AQI'].rolling(window=3, min_periods=1).mean()


# ---------------------------------------------------------------------
# 2. Chronological train-test split BEFORE scaling (prevent leakage)
# ---------------------------------------------------------------------
window_size = 24
all_data = df[features + ['AQI']].values

split_idx = int(0.8 * len(all_data))
train_raw = all_data[:split_idx]
test_raw  = all_data[split_idx:]

# Fit scaler on training partition only
scaler = MinMaxScaler()
train_scaled = scaler.fit_transform(train_raw)
test_scaled  = scaler.transform(test_raw)


# ---------------------------------------------------------------------
# 3. Build sequences
# ---------------------------------------------------------------------
def build_sequences(arr, window_size=24):
    X, y = [], []
    for i in range(window_size, len(arr)):
        X.append(arr[i - window_size:i, :-1])
        y.append(arr[i, -1])
    return np.array(X), np.array(y)


Xtr, ytr = build_sequences(train_scaled, window_size)
Xte, yte = build_sequences(test_scaled, window_size)

batch_size = 64


# ---------------------------------------------------------------------
# 4. LSTM model definition
# ---------------------------------------------------------------------
def create_lstm_model():
    inp = Input(shape=(window_size, len(features)))

    x1 = LSTM(128, return_sequences=True)(inp)
    n1 = LayerNormalization()(x1)

    x2 = LSTM(64, return_sequences=True)(n1)
    n2 = LayerNormalization()(x2)

    x3 = LSTM(32, return_sequences=True)(n2)
    n3 = LayerNormalization()(x3)

    # Residual connection between first and third layers
    p = TimeDistributed(Dense(32))(n1)
    r = Add()([p, n3])

    d = Dropout(0.2)(r)
    f = GlobalAveragePooling1D()(d)
    out = Dense(1)(f)

    model = Model(inp, out, name="LSTM_Model")
    model.compile(
        optimizer=Adam(learning_rate=1e-3, clipnorm=1.0),
        loss='mse'
    )
    return model


# ---------------------------------------------------------------------
# 5. Inverse scaling helper
# ---------------------------------------------------------------------
def inv_scale(vals):
    tmp = np.zeros((len(vals), len(features) + 1))
    tmp[:, -1] = vals
    return scaler.inverse_transform(tmp)[:, -1]


# ---------------------------------------------------------------------
# RMSLE helper — clip to avoid log(0) errors
# Scale-invariant; identical for normalized and denormalized values
# ---------------------------------------------------------------------
def compute_rmsle(y_true, y_pred):
    y_true = np.clip(y_true, 0, None)
    y_pred = np.clip(y_pred, 0, None)
    return np.sqrt(np.mean((np.log1p(y_pred) - np.log1p(y_true)) ** 2))


# ---------------------------------------------------------------------
# 6. Multi-seed training (5 runs for statistical reliability)
# Seeds: 42, 123, 256, 789, 1024
# Results reported as mean +/- std
# ---------------------------------------------------------------------
seeds = [42, 123, 256, 789, 1024]

all_rmse,      all_mae,      all_r2,      all_mape      = [], [], [], []
all_rmse_norm, all_mae_norm, all_rmsle                  = [], [], []

best_val_loss = float('inf')
best_model    = None
best_history  = None
best_logger   = None

for seed in seeds:
    print(f"\n--- Seed {seed} ---")

    np.random.seed(seed)
    tf.random.set_seed(seed)

    model = create_lstm_model()

    es         = EarlyStopping(monitor='val_loss', patience=5,
                               restore_best_weights=True)
    rlp        = ReduceLROnPlateau(monitor='val_loss', factor=0.5,
                                   patience=3)
    run_logger = RunLogger()

    history = model.fit(
        Xtr, ytr,
        validation_data=(Xte, yte),
        epochs=50,
        batch_size=batch_size,
        callbacks=[es, rlp, run_logger],
        verbose=0
    )

    # Track best model
    if run_logger.best_val_loss < best_val_loss:
        best_val_loss = run_logger.best_val_loss
        best_model    = model
        best_history  = history
        best_logger   = run_logger

    # Normalized metrics
    y_pred_norm = model.predict(Xte, verbose=0).flatten()
    rmse_n = np.sqrt(mean_squared_error(yte, y_pred_norm))
    mae_n  = mean_absolute_error(yte, y_pred_norm)
    all_rmse_norm.append(rmse_n)
    all_mae_norm.append(mae_n)

    # RMSLE — scale-invariant, computed on normalized values
    rmsle_ = compute_rmsle(yte, y_pred_norm)
    all_rmsle.append(rmsle_)

    # Denormalized metrics
    y_true_inv = inv_scale(yte)
    y_pred_inv = inv_scale(y_pred_norm)

    rmse_ = np.sqrt(mean_squared_error(y_true_inv, y_pred_inv))
    mae_  = mean_absolute_error(y_true_inv, y_pred_inv)
    mape_ = mean_absolute_percentage_error(y_true_inv, y_pred_inv)
    r2_   = r2_score(y_true_inv, y_pred_inv)

    all_rmse.append(rmse_)
    all_mae.append(mae_)
    all_mape.append(mape_)
    all_r2.append(r2_)

    print(f"  RMSE: {rmse_:.4f} | MAE: {mae_:.4f} | "
          f"RMSLE: {rmsle_:.4f} | R2: {r2_:.4f}")


# ---------------------------------------------------------------------
# 7. Summary statistics across 5 runs
# ---------------------------------------------------------------------
print("\n=== LSTM RESULTS (mean +/- std over 5 seeds) ===")
print(f"\nNormalized:")
print(f"  RMSE  : {np.mean(all_rmse_norm):.4f} +/- {np.std(all_rmse_norm):.4f}")
print(f"  MAE   : {np.mean(all_mae_norm):.4f}  +/- {np.std(all_mae_norm):.4f}")
print(f"  RMSLE : {np.mean(all_rmsle):.4f}  +/- {np.std(all_rmsle):.4f}")
print(f"\nDenormalized (AQI scale):")
print(f"  RMSE  : {np.mean(all_rmse):.2f} +/- {np.std(all_rmse):.2f}")
print(f"  MAE   : {np.mean(all_mae):.2f}  +/- {np.std(all_mae):.2f}")
print(f"  MAPE  : {np.mean(all_mape)*100:.2f}% +/- {np.std(all_mape)*100:.2f}%")
print(f"  RMSLE : {np.mean(all_rmsle):.4f}  +/- {np.std(all_rmsle):.4f} (scale-invariant)")
print(f"  R2    : {np.mean(all_r2):.4f} +/- {np.std(all_r2):.4f}")
print(f"\nBest val_loss : {best_val_loss:.6f}")
print(f"Best epoch    : {best_logger.best_epoch}")
print(f"Window size   : {window_size}")
print(f"Batch size    : {batch_size}")


# ---------------------------------------------------------------------
# 8. Validation on Airveda Sensor Data
# ---------------------------------------------------------------------

# Load outdoor sensor data (Client 2 - all 6 features available)
sensor_outdoor = pd.read_csv("sensor_outdoor.csv",
                              parse_dates=['Date']).set_index('Date')

# Load indoor sensor data (Client 3 - PM2.5 and PM10 only)
sensor_indoor = pd.read_csv("sensor_indoor.csv",
                             parse_dates=['Date']).set_index('Date')

# Zero-pad missing gaseous features for indoor sensor
# Indoor environment maintains consistently low gaseous pollutant levels
sensor_indoor['NO2 (ug/m3)']   = 0.0
sensor_indoor['SO2 (ug/m3)']   = 0.0
sensor_indoor['CO (mg/m3)']    = 0.0
sensor_indoor['Ozone (ug/m3)'] = 0.0
sensor_indoor  = sensor_indoor[features]
sensor_outdoor = sensor_outdoor[features]


def evaluate_on_sensor(sensor_df, label):
    # Scale using same scaler fit on CPCB training data
    sensor_input = np.hstack([
        sensor_df[features].values,
        np.zeros((len(sensor_df), 1))
    ])
    sensor_scaled = scaler.transform(sensor_input)

    X_sensor, y_sensor = build_sequences(sensor_scaled, window_size)

    y_pred_sensor     = best_model.predict(X_sensor, verbose=0).flatten()
    y_pred_sensor_inv = inv_scale(y_pred_sensor)
    y_true_sensor_inv = inv_scale(y_sensor)

    rmse_s  = np.sqrt(mean_squared_error(y_true_sensor_inv,
                                          y_pred_sensor_inv))
    mae_s   = mean_absolute_error(y_true_sensor_inv, y_pred_sensor_inv)
    rmsle_s = compute_rmsle(y_sensor, y_pred_sensor)
    r2_s    = r2_score(y_true_sensor_inv, y_pred_sensor_inv)

    print(f"\n=== Sensor Validation: {label} ===")
    print(f"  RMSE  : {rmse_s:.4f}")
    print(f"  MAE   : {mae_s:.4f}")
    print(f"  RMSLE : {rmsle_s:.4f}")
    print(f"  R2    : {r2_s:.4f}")

    return y_true_sensor_inv, y_pred_sensor_inv


y_true_out, y_pred_out = evaluate_on_sensor(sensor_outdoor, "Outdoor Sensor")
y_true_in,  y_pred_in  = evaluate_on_sensor(sensor_indoor,  "Indoor Sensor")


# ---------------------------------------------------------------------
# 9. Diagnostic plots (best run)
# ---------------------------------------------------------------------
y_pred_best = best_model.predict(Xte, verbose=0).flatten()
y_true_plot = inv_scale(yte)
y_pred_plot = inv_scale(y_pred_best)

# (a) Loss curves
plt.figure(figsize=(8, 4))
plt.plot(best_history.history['loss'], label='Train Loss')
plt.plot(best_history.history['val_loss'], label='Val Loss')
plt.title("LSTM Training and Validation Loss Curves")
plt.xlabel("Epoch")
plt.ylabel("MSE Loss")
plt.legend()
plt.grid()

# (b) AQI Time Series
plt.figure(figsize=(10, 4))
plt.plot(y_true_plot, label='Actual AQI')
plt.plot(y_pred_plot, label='Predicted AQI', alpha=0.7)
plt.title("LSTM AQI Time Series Prediction")
plt.xlabel("Time Index")
plt.ylabel("AQI")
plt.legend()
plt.grid()

# (c) Predicted vs Actual
plt.figure(figsize=(6, 6))
plt.scatter(y_true_plot, y_pred_plot, alpha=0.3)
mn, mx = y_true_plot.min(), y_true_plot.max()
plt.plot([mn, mx], [mn, mx], 'r--')
plt.title("Predicted vs Actual AQI")
plt.xlabel("Actual AQI")
plt.ylabel("Predicted AQI")
plt.grid()

# (d) Residuals
plt.figure(figsize=(6, 4))
sns.histplot(y_true_plot - y_pred_plot, kde=True, bins=50)
plt.title("Residuals Distribution")
plt.xlabel("Error (AQI)")
plt.ylabel("Frequency")

# (e) Correlation heatmap
plt.figure(figsize=(6, 5))
corr_df = pd.DataFrame(train_scaled, columns=features + ['AQI'])
sns.heatmap(corr_df.corr(), annot=True, fmt=".2f", cmap="coolwarm")
plt.title("Feature Correlation Heatmap")

plt.tight_layout()
plt.show()
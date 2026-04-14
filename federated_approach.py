import numpy as np
import pandas as pd
import time
import matplotlib.pyplot as plt
from sklearn.preprocessing import MinMaxScaler
from sklearn.metrics import (
    mean_absolute_error, mean_squared_error, r2_score,
    mean_absolute_percentage_error
)
from tensorflow.keras.models import Model
from tensorflow.keras.layers import (
    Input, LSTM, Bidirectional, LayerNormalization, Dense,
    TimeDistributed, Add, Dropout, GlobalAveragePooling1D, Attention
)
from tensorflow.keras.optimizers import Adam
from tensorflow.keras.callbacks import EarlyStopping
import tensorflow as tf
from phe import paillier


# ---------------------------------------------------------------------
# 1. Load CPCB data (Client 1)
# ---------------------------------------------------------------------
df = pd.read_csv("jld_aqi_with_aqi.csv", parse_dates=['Date'])
df = df.set_index('Date')

features = [
    'PM2.5 (ug/m3)', 'PM10 (ug/m3)', 'NO2 (ug/m3)',
    'SO2 (ug/m3)', 'CO (mg/m3)', 'Ozone (ug/m3)'
]

# 3-step rolling average prior to split — backward-looking, no leakage
df['AQI'] = df['AQI'].rolling(window=3, min_periods=1).mean()

# Chronological split BEFORE scaling
split_idx  = int(0.8 * len(df))
train_raw  = df[features + ['AQI']].values[:split_idx]
test_raw   = df[features + ['AQI']].values[split_idx:]

# Fit scaler on training partition only
scaler_c1 = MinMaxScaler()
train_c1  = scaler_c1.fit_transform(train_raw)
test_c1   = scaler_c1.transform(test_raw)


# ---------------------------------------------------------------------
# 2. Load Sensor Data (Client 2 — outdoor, Client 3 — indoor)
# ---------------------------------------------------------------------
sensor_outdoor = pd.read_csv("sensor_outdoor.csv",
                              parse_dates=['Date']).set_index('Date')
sensor_indoor  = pd.read_csv("sensor_indoor.csv",
                              parse_dates=['Date']).set_index('Date')

# Zero-pad missing gaseous features for indoor sensor
# Indoor environment maintains consistently low gaseous pollutant levels
sensor_indoor['NO2 (ug/m3)']   = 0.0
sensor_indoor['SO2 (ug/m3)']   = 0.0
sensor_indoor['CO (mg/m3)']    = 0.0
sensor_indoor['Ozone (ug/m3)'] = 0.0

# Scale sensor data using CPCB scaler (same feature space)
def scale_sensor(sensor_df):
    arr = np.hstack([
        sensor_df[features].values,
        np.zeros((len(sensor_df), 1))
    ])
    return scaler_c1.transform(arr)

scaled_c2 = scale_sensor(sensor_outdoor)
scaled_c3 = scale_sensor(sensor_indoor)


# ---------------------------------------------------------------------
# 3. Build sequences
# ---------------------------------------------------------------------
window_size = 24

def build_sequences(arr, window_size=24):
    X, y = [], []
    for i in range(window_size, len(arr)):
        X.append(arr[i - window_size:i, :-1])
        y.append(arr[i, -1])
    return np.array(X), np.array(y)


# Client datasets
Xtr_c1, ytr_c1 = build_sequences(train_c1, window_size)
Xte_c1, yte_c1 = build_sequences(test_c1,  window_size)
X_c2,   y_c2   = build_sequences(scaled_c2, window_size)
X_c3,   y_c3   = build_sequences(scaled_c3, window_size)

# Sample counts for proportional FedAvg
# Client 1 ~90%, Client 2 ~5%, Client 3 ~5%
n1 = len(Xtr_c1)
n2 = len(X_c2)
n3 = len(X_c3)
n_total = n1 + n2 + n3

client_data = [
    (Xtr_c1, ytr_c1, n1),
    (X_c2,   y_c2,   n2),
    (X_c3,   y_c3,   n3),
]

print(f"Client 1 (CPCB)    : {n1} samples ({n1/n_total*100:.1f}%)")
print(f"Client 2 (Outdoor) : {n2} samples ({n2/n_total*100:.1f}%)")
print(f"Client 3 (Indoor)  : {n3} samples ({n3/n_total*100:.1f}%)")


# ---------------------------------------------------------------------
# 4. Inverse scaling helper
# ---------------------------------------------------------------------
def inv_scale(vals):
    tmp = np.zeros((len(vals), len(features) + 1))
    tmp[:, -1] = vals
    return scaler_c1.inverse_transform(tmp)[:, -1]


# ---------------------------------------------------------------------
# RMSLE helper
# ---------------------------------------------------------------------
def compute_rmsle(y_true, y_pred):
    y_true = np.clip(y_true, 0, None)
    y_pred = np.clip(y_pred, 0, None)
    return np.sqrt(np.mean((np.log1p(y_pred) - np.log1p(y_true)) ** 2))


# ---------------------------------------------------------------------
# 5. Paillier HE setup
# Encryption applied to output Dense layer weights only
# Full-model HE is computationally infeasible for edge deployment
# ---------------------------------------------------------------------
public_key, private_key = paillier.generate_paillier_keypair(n_length=1024)
SAMPLING_RATIO = 0.10  # 10% sparse sampling of Dense kernel


def encrypt_dense_weights(weights, pub_key, ratio):
    """Encrypt a random subset of Dense layer weights."""
    flat   = weights.flatten().tolist()
    n      = len(flat)
    k      = max(1, int(n * ratio))
    idx    = np.random.choice(n, k, replace=False)
    enc    = {i: pub_key.encrypt(flat[i]) for i in idx}
    return flat, enc, idx, n


def decrypt_and_aggregate(enc_list, idx_list, flat_list,
                           shapes, priv_key, weights_list):
    """
    Aggregate encrypted Dense kernel values using HE addition,
    then decrypt and average. Remaining weights averaged in plaintext.
    """
    # Plaintext average for non-sampled indices
    avg_flat = np.mean([f for f in flat_list], axis=0)

    # HE aggregation for sampled indices
    for i_enc, (enc, idx) in enumerate(zip(enc_list, idx_list)):
        for pos in idx:
            if i_enc == 0:
                agg = enc[pos]
            else:
                agg = agg + enc[pos]
        if len(idx_list) > 0:
            dec_val = priv_key.decrypt(agg) / len(enc_list)
            avg_flat[pos] = dec_val

    return avg_flat.reshape(shapes)


# ---------------------------------------------------------------------
# 6. Model definitions
# ---------------------------------------------------------------------
def create_lstm_model():
    inp = Input(shape=(window_size, len(features)))
    x1  = LSTM(128, return_sequences=True)(inp)
    n1  = LayerNormalization()(x1)
    x2  = LSTM(64, return_sequences=True)(n1)
    n2  = LayerNormalization()(x2)
    x3  = LSTM(32, return_sequences=True)(n2)
    n3  = LayerNormalization()(x3)
    p   = TimeDistributed(Dense(32))(n1)
    r   = Add()([p, n3])
    d   = Dropout(0.2)(r)
    f   = GlobalAveragePooling1D()(d)
    out = Dense(1)(f)
    m   = Model(inp, out, name="LSTM_FL")
    m.compile(optimizer=Adam(learning_rate=1e-3, clipnorm=1.0), loss='mse')
    return m


def create_bilstm_model():
    inp      = Input(shape=(window_size, len(features)))
    x1       = Bidirectional(LSTM(128, return_sequences=True))(inp)
    x1       = LayerNormalization()(x1)
    x2       = Bidirectional(LSTM(64, return_sequences=True))(x1)
    x2       = LayerNormalization()(x2)
    x3       = Bidirectional(LSTM(32, return_sequences=True))(x2)
    x3       = LayerNormalization()(x3)
    proj     = TimeDistributed(Dense(64))(x1)
    residual = Add()([proj, x3])
    residual = Dropout(0.2)(residual)
    att      = Attention()([residual, residual])
    att      = Dropout(0.2)(att)
    flat     = GlobalAveragePooling1D()(att)
    out      = Dense(1)(flat)
    m        = Model(inp, out, name="BiLSTM_FL")
    m.compile(optimizer=Adam(learning_rate=1e-3, clipnorm=1.0), loss='mse')
    return m


# ---------------------------------------------------------------------
# 7. Federated Training with Paillier HE
# ---------------------------------------------------------------------
def run_federated(create_fn, model_name, rounds=5, local_epochs=3):
    print(f"\n{'='*60}")
    print(f" Federated Training: {model_name}")
    print(f"{'='*60}")

    global_model = create_fn()

    round_r2      = []
    round_avg_sec = []
    round_total_sec = []
    cumulative = 0.0

    for rnd in range(rounds):
        print(f"\n--- Round {rnd + 1} ---")
        round_start = time.time()

        local_weights_list = []
        local_losses       = []
        enc_list           = []
        idx_list           = []
        flat_list          = []

        for X_c, y_c, n_c in client_data:
            local_model = create_fn()
            local_model.set_weights(global_model.get_weights())

            local_model.fit(
                X_c, y_c,
                epochs=local_epochs,
                batch_size=64,
                verbose=0
            )

            w = local_model.get_weights()

            # Paillier HE on output Dense layer (last kernel weight)
            dense_kernel = w[-2]  # second-to-last = Dense kernel
            flat, enc, idx, _ = encrypt_dense_weights(
                dense_kernel, public_key, SAMPLING_RATIO
            )
            enc_list.append(enc)
            idx_list.append(idx)
            flat_list.append(flat)

            local_weights_list.append(w)
            local_losses.append(
                local_model.evaluate(X_c, y_c, verbose=0)
            )

        # Sample-proportional FedAvg for all layers except Dense kernel
        new_weights = []
        for layer_idx in range(len(local_weights_list[0])):
            if layer_idx == len(local_weights_list[0]) - 2:
                # Dense kernel — aggregate under HE
                agg = decrypt_and_aggregate(
                    enc_list, idx_list, flat_list,
                    local_weights_list[0][layer_idx].shape,
                    private_key, local_weights_list
                )
                new_weights.append(agg)
            else:
                # Plaintext sample-proportional FedAvg
                agg = sum(
                    (n_c / n_total) * w[layer_idx]
                    for (_, _, n_c), w in zip(client_data,
                                               local_weights_list)
                )
                new_weights.append(agg)

        global_model.set_weights(new_weights)

        # Evaluate on CPCB test set
        y_pred_rnd = global_model.predict(Xte_c1, verbose=0).flatten()
        r2_rnd     = r2_score(yte_c1, y_pred_rnd)

        round_time = time.time() - round_start
        cumulative += round_time
        round_avg_sec.append(round_time)
        round_total_sec.append(cumulative)
        round_r2.append(r2_rnd)

        avg_loss = np.mean(local_losses)
        print(f"  Avg Loss : {avg_loss:.5f} | R2 : {r2_rnd:.4f} | "
              f"Time : {round_time:.1f}s (cumulative: {cumulative:.1f}s)")

    return global_model, round_r2, round_avg_sec, round_total_sec


# Run for both architectures
lstm_model,   lstm_r2,   lstm_avg,   lstm_total   = run_federated(
    create_lstm_model,   "LSTM + Paillier")
bilstm_model, bilstm_r2, bilstm_avg, bilstm_total = run_federated(
    create_bilstm_model, "BiLSTM + Paillier")


# ---------------------------------------------------------------------
# 8. Final evaluation
# ---------------------------------------------------------------------
def evaluate_global(model, model_name):
    y_pred_norm = model.predict(Xte_c1, verbose=0).flatten()
    y_true_inv  = inv_scale(yte_c1)
    y_pred_inv  = inv_scale(y_pred_norm)

    rmse  = np.sqrt(mean_squared_error(y_true_inv, y_pred_inv))
    mae   = mean_absolute_error(y_true_inv, y_pred_inv)
    mape  = mean_absolute_percentage_error(y_true_inv, y_pred_inv)
    rmsle = compute_rmsle(yte_c1, y_pred_norm)
    r2    = r2_score(y_true_inv, y_pred_inv)

    print(f"\n=== {model_name} Final Results (CPCB Test Set) ===")
    print(f"  RMSE  : {rmse:.4f}")
    print(f"  MAE   : {mae:.4f}")
    print(f"  MAPE  : {mape*100:.2f}%")
    print(f"  RMSLE : {rmsle:.4f} (scale-invariant)")
    print(f"  R2    : {r2:.4f}")

    return y_true_inv, y_pred_inv


y_true_lstm,   y_pred_lstm   = evaluate_global(lstm_model,   "FL + LSTM + HE")
y_true_bilstm, y_pred_bilstm = evaluate_global(bilstm_model, "FL + BiLSTM + HE")


# ---------------------------------------------------------------------
# 9. Per-round summary table
# ---------------------------------------------------------------------
print("\n=== Per-Round Performance ===")
print(f"{'R':<4} {'LSTM Avg(s)':<14} {'LSTM Total(s)':<16} "
      f"{'LSTM R2':<12} {'BiLSTM Avg(s)':<16} "
      f"{'BiLSTM Total(s)':<18} {'BiLSTM R2'}")
for i in range(5):
    print(f"{i+1:<4} {lstm_avg[i]:<14.1f} {lstm_total[i]:<16.1f} "
          f"{lstm_r2[i]:<12.4f} {bilstm_avg[i]:<16.1f} "
          f"{bilstm_total[i]:<18.1f} {bilstm_r2[i]:.4f}")


# ---------------------------------------------------------------------
# 10. Plots
# ---------------------------------------------------------------------

# (a) R2 convergence per round — LSTM
fig, axes = plt.subplots(1, 2, figsize=(12, 4))

axes[0].plot(range(1, 6), lstm_r2, marker='o', label='R2')
ax2 = axes[0].twinx()
ax2.bar(range(1, 6), lstm_total, alpha=0.3, color='orange',
        label='Cumulative Time (s)')
axes[0].set_title("LSTM + Paillier: R2 Convergence & Overhead")
axes[0].set_xlabel("Round")
axes[0].set_ylabel("R2")
ax2.set_ylabel("Cumulative Time (s)")
axes[0].legend(loc='lower right')

# (b) R2 convergence per round — BiLSTM
axes[1].plot(range(1, 6), bilstm_r2, marker='o',
             color='green', label='R2')
ax3 = axes[1].twinx()
ax3.bar(range(1, 6), bilstm_total, alpha=0.3, color='orange',
        label='Cumulative Time (s)')
axes[1].set_title("BiLSTM + Paillier: R2 Convergence & Overhead")
axes[1].set_xlabel("Round")
axes[1].set_ylabel("R2")
ax3.set_ylabel("Cumulative Time (s)")
axes[1].legend(loc='lower right')

plt.tight_layout()
plt.show()

# (c) AQI Time Series — FL + BiLSTM
plt.figure(figsize=(10, 4))
plt.plot(y_true_bilstm, label='Actual AQI')
plt.plot(y_pred_bilstm, label='FL + BiLSTM + HE Predicted', alpha=0.7)
plt.title("FL + BiLSTM + Paillier: AQI Time Series")
plt.xlabel("Time Index")
plt.ylabel("AQI")
plt.legend()
plt.grid()
plt.show()
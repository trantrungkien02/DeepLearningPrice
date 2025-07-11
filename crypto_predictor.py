import pandas as pd
import numpy as np
import tensorflow as tf
from tensorflow.keras.models import Model
from tensorflow.keras.layers import Input, LSTM, Dropout, Dense
from sklearn.preprocessing import MinMaxScaler

# Load dữ liệu từ file
df = pd.read_csv("recent_btc_eth_prices.csv")
df = df[['btc_close', 'eth_close']]

# Chuẩn hóa dữ liệu
scaler = MinMaxScaler()
scaled = scaler.fit_transform(df)

# Tạo chuỗi dữ liệu
def create_sequences(data, seq_len=24):
    X, y_btc, y_eth = [], [], []
    for i in range(len(data) - seq_len - 1):
        X.append(data[i:i+seq_len])
        y_btc.append(int(data[i+seq_len][0] > data[i+seq_len-1][0]))
        y_eth.append(int(data[i+seq_len][1] > data[i+seq_len-1][1]))
    return np.array(X), np.array(y_btc), np.array(y_eth)

X, y_btc, y_eth = create_sequences(scaled, seq_len=24)

# Chia dữ liệu
split = int(0.8 * len(X))
X_train, X_test = X[:split], X[split:]
y_btc_train, y_btc_test = y_btc[:split], y_btc[split:]
y_eth_train, y_eth_test = y_eth[:split], y_eth[split:]

# Mô hình LSTM 2 đầu ra (BTC & ETH)
inp = Input(shape=(24, 2))
x = LSTM(64, return_sequences=True)(inp)
x = Dropout(0.2)(x)
x = LSTM(32)(x)
x = Dropout(0.2)(x)
btc_out = Dense(1, activation='sigmoid', name='btc_output')(x)
eth_out = Dense(1, activation='sigmoid', name='eth_output')(x)

model = Model(inputs=inp, outputs=[btc_out, eth_out])
model.compile(optimizer='adam', loss='binary_crossentropy', metrics=['accuracy'])

# Huấn luyện
model.fit(X_train, [y_btc_train, y_eth_train], epochs=10, batch_size=16, validation_split=0.1)

# Dự đoán
latest_seq = scaled[-24:].reshape(1, 24, 2)
btc_pred, eth_pred = model.predict(latest_seq)
btc_prob = btc_pred[0][0]
eth_prob = eth_pred[0][0]

btc_trend = "↑ Increase" if btc_prob > 0.5 else "↓ Decrease"
eth_trend = "↑ Increase" if eth_prob > 0.5 else "↓ Decrease"

btc_now = df['btc_close'].iloc[-1]
eth_now = df['eth_close'].iloc[-1]

btc_pct = round((btc_prob - 0.5) * 4 * 100, 2)
eth_pct = round((eth_prob - 0.5) * 4 * 100, 2)

btc_pred_price = round(btc_now * (1 + btc_pct / 100), 2)
eth_pred_price = round(eth_now * (1 + eth_pct / 100), 2)

print(f"\n🔮 BTC: {btc_trend} | Change: {btc_pct}% | Predicted: ${btc_pred_price}")
print(f"🔮 ETH: {eth_trend} | Change: {eth_pct}% | Predicted: ${eth_pred_price}")

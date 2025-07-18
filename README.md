# 🧠 Crypto Price Trend Predictor (BTC & ETH)

A TensorFlow-based deep learning model that predicts the **up/down trend** of **Bitcoin (BTC)** and **Ethereum (ETH)** using past 24-day historical prices.

---

## 🚀 Features

- 📊 Uses LSTM (Long Short-Term Memory) to detect price trends
- 🔮 Predicts:
  - Price direction (↑ Increase / ↓ Decrease)
  - Estimated percentage change
  - Predicted price for the next day
- 🧠 Trains simultaneously on both BTC and ETH (multi-output model)

---

## 📁 Input CSV Format

The project requires a CSV file named `recent_btc_eth_prices.csv` with the following structure:

```csv
date,btc_close,eth_close
2025-04-01,62900.00,3520.00
2025-04-02,63120.50,3545.20
...
```

## 🛠️ Installation

Install the required Python libraries:

```bash
pip install pandas numpy tensorflow scikit-learn
```

## ▶️ How to Run
Make sure your script is named something like crypto_dual_predictor.py, then run:

```bash
python crypto_dual_predictor.py
```

## 📤 Sample Output

```text
🔮 BTC: ↑ Increase | Change: +1.72% | Predicted: $63,800.40  
🔮 ETH: ↓ Decrease | Change: -0.44% | Predicted: $3,392.50
```

## 🧠 How It Works

Input: Last 24 days of historical BTC & ETH prices
Model:
- 2 LSTM layers shared for both assets
- 2 output heads:

BTC: probability of price increase
ETH: probability of price increase

Post-processing:
- Probabilities converted to direction (↑ / ↓)
- Estimated percentage change calculated from probability

Final predicted price computed from last known value

## 📌 Future Enhancements
- 🔗 Fetch live price data from CoinGecko or Binance API

- 💾 Save & load model weights using .h5

- 🌐 Build a REST API using Flask or FastAPI

- 🧩 Integrate real-time predictions into a Web3 dashboard

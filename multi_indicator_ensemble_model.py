# multi_indicator_ensemble_model.py

import pandas as pd
import numpy as np
import yfinance as yf
from sklearn.ensemble import RandomForestClassifier
from sklearn.model_selection import TimeSeriesSplit, cross_val_score
from sklearn.metrics import accuracy_score, f1_score

# ---------------------------
# 1. Data Extraction (replace with SQL extraction in prod)
# ---------------------------
symbol = "^GSPC"  # S&P 500 index
start_date = "2019-01-01"
end_date = "2021-12-31"

data = yf.download(symbol, start=start_date, end=end_date)
data['Return'] = data['Adj Close'].pct_change()

# ---------------------------
# 2. Feature Engineering
# ---------------------------
# EMA crossovers
data['EMA10'] = data['Adj Close'].ewm(span=10, adjust=False).mean()
data['EMA50'] = data['Adj Close'].ewm(span=50, adjust=False).mean()
data['EMA_Cross'] = np.where(data['EMA10'] > data['EMA50'], 1, 0)

# RSI
def compute_rsi(series, period=14):
    delta = series.diff()
    gain = np.where(delta > 0, delta, 0)
    loss = np.where(delta < 0, -delta, 0)
    avg_gain = pd.Series(gain).rolling(window=period).mean()
    avg_loss = pd.Series(loss).rolling(window=period).mean()
    rs = avg_gain / avg_loss
    rsi = 100 - (100 / (1 + rs))
    return rsi

data['RSI'] = compute_rsi(data['Adj Close'])

# MACD Histogram
ema12 = data['Adj Close'].ewm(span=12, adjust=False).mean()
ema26 = data['Adj Close'].ewm(span=26, adjust=False).mean()
data['MACD'] = ema12 - ema26
data['Signal'] = data['MACD'].ewm(span=9, adjust=False).mean()
data['MACD_Hist'] = data['MACD'] - data['Signal']

# Bollinger Band Width
data['SMA20'] = data['Adj Close'].rolling(window=20).mean()
data['STD20'] = data['Adj Close'].rolling(window=20).std()
data['BB_Width'] = (data['SMA20'] + 2*data['STD20'] - (data['SMA20'] - 2*data['STD20'])) / data['SMA20']

# Lagged returns
data['Return_1'] = data['Return'].shift(1)
data['Return_2'] = data['Return'].shift(2)

# Drop NaNs
data.dropna(inplace=True)

# ---------------------------
# 3. Target Construction
# ---------------------------
data['Target'] = np.where(data['Return'].shift(-1) > 0, 1, 0)

# ---------------------------
# 4. Features & Train/Test Split
# ---------------------------
features = ['EMA_Cross', 'RSI', 'MACD_Hist', 'BB_Width', 'Return_1', 'Return_2']
X = data[features]
y = data['Target']

# TimeSeriesSplit for cross-validation
tscv = TimeSeriesSplit(n_splits=5)

# ---------------------------
# 5. Random Forest Model
# ---------------------------
model = RandomForestClassifier(n_estimators=100, max_depth=4, random_state=42)
scores = cross_val_score(model, X, y, cv=tscv, scoring='f1')
print("F1 scores (CV):", scores)
print("Mean F1 score:", scores.mean())

# Train & evaluate on last 20% as test
split_idx = int(len(X) * 0.8)
X_train, X_test = X[:split_idx], X[split_idx:]
y_train, y_test = y[:split_idx], y[split_idx:]

model.fit(X_train, y_train)
y_pred = model.predict(X_test)

accuracy = accuracy_score(y_test, y_pred)
f1 = f1_score(y_test, y_pred)

print("Test Accuracy:", accuracy)
print("Test F1 Score:", f1)

# ---------------------------
# 6. Signal & Backtest
# ---------------------------
data['Signal'] = model.predict(data[features])
data['Strategy_Return'] = data['Return'] * data['Signal'].shift(1)
data['Cumulative_Strategy'] = (1 + data['Strategy_Return']).cumprod()
data['Cumulative_Market'] = (1 + data['Return']).cumprod()

import matplotlib.pyplot as plt
plt.plot(data['Cumulative_Strategy'], label='Strategy')
plt.plot(data['Cumulative_Market'], label='Market')
plt.legend()
plt.show()

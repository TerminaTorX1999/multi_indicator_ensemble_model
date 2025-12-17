# Multi-Indicator Ensemble Signal Model

This repository contains a Python-based ensemble trading model using multiple technical indicators combined with a Random Forest classifier.

## Strategy Overview
- Indicators used:
  - EMA crossover (10 / 50)
  - RSI
  - MACD histogram
  - Bollinger Band width
  - Lagged returns
- Ensemble learning with Random Forest
- Time-series cross-validation and F1-score evaluation

## Data
- Daily S&P 500 index data (~3 years)
- Data sourced programmatically (can be replaced with SQL or internal feeds)

## Usage
1. Install dependencies: `pandas`, `numpy`, `yfinance`, `scikit-learn`, `matplotlib`
2. Run:

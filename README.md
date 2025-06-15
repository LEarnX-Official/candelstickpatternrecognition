# candelstickpatternrecognition
# 📊 Candlestick Pattern Classification & Trading Signal System

This project simulates candlestick charts, detects patterns using TA-Lib, trains a CNN on generated images, and predicts trading signals (Buy/Sell/Hold) based on chart patterns.

---

## 🏗️ 1. Dataset Generation (`c2.py`)

- Simulates synthetic OHLCV data using Geometric Brownian Motion.
- Detects patterns with **TA-Lib**:
  - Bullish, Bearish, Neutral candlestick types.
- Saves 100 charts per pattern using `mplfinance`.
- Also generates advanced patterns:
  - `cup_and_handle`, `falling_wedge`, `bull_flag`, `head_and_shoulders`, `symmetrical_triangle`

**Output Structure:**
```
dataset/
├── bullish/
├── bearish/
├── neutral/
└── advanced/
    ├── cup_and_handle/
    ├── falling_wedge/
    ├── bull_flag/
    ├── head_and_shoulders/
    └── symmetrical_triangle/
```

**Run:**
```bash
python c2.py
```

---

## 🧠 2. Model Training (`m3.py`)

- Loads and merges:
  - `dataset1/` with labels: `bullish_base`, `bearish_base`, etc.
  - `dataset2/` with sub-patterns like `bullish_cdlhammer`, `bearish_cdlshootingstar`, etc.
- Preprocesses all images to `128x128` size.
- CNN Architecture:
  - 2 Conv layers + MaxPooling
  - Dense → Softmax
- Saves trained model:
  - `trained_combined_model.h5`
  - `combined_label_encoder.pkl`
- Outputs classification metrics and accuracy.

**Run:**
```bash
python m3.py
```

---

## 🤖 3. Trading Inference (`trade.py`)

- Loads the trained model and encoder.
- Takes a candlestick chart image and predicts the pattern.
- Returns:
  - Predicted Pattern Label
  - Confidence Score
  - **Trading Signal**: 
    - Bullish → `Buy`
    - Bearish → `Sell`
    - Neutral → `Hold / No Action`

**Example:**
```bash
python trade.py
```

In `trade.py`, update the image path:
```python
test_image_path = 's.png'  # Replace with your image
```

---

✅ **Dependencies:**
- `tensorflow`, `pandas`, `numpy`, `matplotlib`, `mplfinance`, `talib`, `scikit-learn`, `joblib`

```bash
pip install -r requirements.txt
```

---

📁 Files:
- `c2.py` → Dataset generation
- `m3.py` → CNN training
- `trade.py` → Prediction + trading signal

---

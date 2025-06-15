import os
import pandas as pd
import numpy as np
import mplfinance as mpf
import talib

# === Create dataset directories ===
os.makedirs("dataset/bullish", exist_ok=True)
os.makedirs("dataset/bearish", exist_ok=True)
os.makedirs("dataset/neutral", exist_ok=True)
os.makedirs("dataset/advanced/cup_and_handle", exist_ok=True)
os.makedirs("dataset/advanced/falling_wedge", exist_ok=True)
os.makedirs("dataset/advanced/bull_flag", exist_ok=True)
os.makedirs("dataset/advanced/head_and_shoulders", exist_ok=True)
os.makedirs("dataset/advanced/symmetrical_triangle", exist_ok=True)

# === Helper to safely get TA-Lib pattern function or None ===
def get_talib_func(name):
    return getattr(talib, name, None)

# === TA-Lib pattern functions (only those available in your TA-Lib) ===
all_patterns = {
    "CDLHAMMER": get_talib_func("CDLHAMMER"),
    "CDLINVERTEDHAMMER": get_talib_func("CDLINVERTEDHAMMER"),
    "CDLENGULFING": get_talib_func("CDLENGULFING"),
    "CDLPIERCING": get_talib_func("CDLPIERCING"),
    "CDLMORNINGSTAR": get_talib_func("CDLMORNINGSTAR"),
    "CDL3WHITESOLDIERS": get_talib_func("CDL3WHITESOLDIERS"),
    "CDLHARAMI": get_talib_func("CDLHARAMI"),
    "CDLHARAMICROSS": get_talib_func("CDLHARAMICROSS"),
    "CDLMATCHINGLOW": get_talib_func("CDLMATCHINGLOW"),
    "CDLABANDONEDBABY": get_talib_func("CDLABANDONEDBABY"),
    "CDLKICKING": get_talib_func("CDLKICKING"),
    "CDLMATHOLD": get_talib_func("CDLMATHOLD"),
    "CDLBELTHOLD": get_talib_func("CDLBELTHOLD"),
    "CDLHANGINGMAN": get_talib_func("CDLHANGINGMAN"),
    "CDLSHOOTINGSTAR": get_talib_func("CDLSHOOTINGSTAR"),
    "CDLDARKCLOUDCOVER": get_talib_func("CDLDARKCLOUDCOVER"),
    "CDLEVENINGSTAR": get_talib_func("CDLEVENINGSTAR"),
    "CDL3BLACKCROWS": get_talib_func("CDL3BLACKCROWS"),
}

neutral_patterns = {
    "CDLDOJI": get_talib_func("CDLDOJI"),
    "CDLSPINNINGTOP": get_talib_func("CDLSPINNINGTOP"),
    "CDLGRAVESTONEDOJI": get_talib_func("CDLGRAVESTONEDOJI"),
    "CDLLONGLEGGEDDOJI": get_talib_func("CDLLONGLEGGEDDOJI"),
    "CDLRICKSHAWMAN": get_talib_func("CDLRICKSHAWMAN"),
}

# Remove any None functions (patterns missing in TA-Lib)
for d in (all_patterns, neutral_patterns):
    for k in list(d.keys()):
        if d[k] is None:
            print(f"Warning: Pattern {k} not found in TA-Lib and skipped.")
            del d[k]

# Limits and counters for how many images to generate per pattern
pattern_limit = 100
saved_count = {name: 0 for name in all_patterns}
saved_count.update({name: 0 for name in neutral_patterns})

# === GBM OHLC generator function ===
def simulate_ohlc(n, S0=100, mu=0.0002, sigma=0.01):
    dt = 1/252  # daily step size (1 trading day)
    returns = np.random.normal(loc=mu*dt, scale=sigma*np.sqrt(dt), size=n)
    price = S0 * np.exp(np.cumsum(returns))

    # Generate OHLC prices with small random spreads
    open_ = price * (1 + np.random.normal(0, 0.001, n))
    close = price * (1 + np.random.normal(0, 0.001, n))
    high = np.maximum(open_, close) * (1 + np.random.uniform(0, 0.002, n))
    low = np.minimum(open_, close) * (1 - np.random.uniform(0, 0.002, n))

    df = pd.DataFrame({"Open": open_, "High": high, "Low": low, "Close": close})
    df.index = pd.date_range("2020-01-01", periods=n, freq="H")
    return df

total_target = len(saved_count) * pattern_limit
generated = 0
window_size = 20

print("Starting TA-Lib candlestick pattern dataset generation...")

while generated < total_target:
    df = simulate_ohlc(5000)
    df["Volume"] = np.random.randint(100, 1000, size=len(df))

    for i in range(window_size, len(df)):
        sub_df = df.iloc[i - window_size:i]
        o, h, l, c = sub_df["Open"].values, sub_df["High"].values, sub_df["Low"].values, sub_df["Close"].values

        # Bullish / Bearish patterns
        for name, func in all_patterns.items():
            if saved_count[name] >= pattern_limit:
                continue

            out = func(o, h, l, c)
            val = out[-1]

            if val > 0:
                # Skip bearish-only patterns for bullish signal
                if name in ["CDLHANGINGMAN", "CDLSHOOTINGSTAR", "CDLDARKCLOUDCOVER",
                            "CDLEVENINGSTAR", "CDL3BLACKCROWS"]:
                    continue

                path = f"dataset/bullish/{name.lower()}_{saved_count[name]}.png"
                mpf.plot(sub_df, type='candle', style='charles', savefig=path)
                saved_count[name] += 1
                generated += 1
                break

            elif val < 0:
                # Skip bullish-only patterns for bearish signal
                if name in ["CDLPIERCING", "CDLMORNINGSTAR", "CDL3WHITESOLDIERS",
                            "CDLMATCHINGLOW", "CDLHAMMER", "CDLINVERTEDHAMMER"]:
                    continue

                path = f"dataset/bearish/{name.lower()}_{saved_count[name]}.png"
                mpf.plot(sub_df, type='candle', style='charles', savefig=path)
                saved_count[name] += 1
                generated += 1
                break

        if generated >= total_target:
            break

        # Neutral patterns
        for name, func in neutral_patterns.items():
            if saved_count[name] >= pattern_limit:
                continue

            out = func(o, h, l, c)
            val = out[-1]

            if val != 0:
                path = f"dataset/neutral/{name.lower()}_{saved_count[name]}.png"
                mpf.plot(sub_df, type='candle', style='charles', savefig=path)
                saved_count[name] += 1
                generated += 1
                break

        if generated >= total_target:
            break

print("✅ TA-Lib pattern dataset generation complete.")

# === Custom advanced chart pattern generators ===
def generate_cup_and_handle():
    x = np.linspace(0, 2 * np.pi, 100)
    cup = -np.cos(x) + 2
    handle = np.linspace(cup[-1], cup[-1] * 0.98, 20)
    prices = np.concatenate([cup, handle]) + 100
    noise = np.random.normal(0, 0.05, len(prices))
    close = prices + noise
    open_ = close + np.random.normal(0, 0.1, len(close))
    high = np.maximum(open_, close) + 0.2
    low = np.minimum(open_, close) - 0.2
    df = pd.DataFrame({"Open": open_, "High": high, "Low": low, "Close": close})
    df["Volume"] = np.random.randint(100, 1000, len(close))
    df.index = pd.date_range("2021-01-01", periods=len(df), freq="H")
    return df

def generate_falling_wedge():
    base = np.linspace(100, 90, 100) + np.random.normal(0, 0.3, 100)
    upper = base + np.linspace(2, 0.5, 100)
    lower = base - np.linspace(2, 0.5, 100)
    open_ = base + np.random.normal(0, 0.2, 100)
    close = base + np.random.normal(0, 0.2, 100)
    df = pd.DataFrame({"Open": open_, "High": upper, "Low": lower, "Close": close})
    df["Volume"] = np.random.randint(100, 1000, len(df))
    df.index = pd.date_range("2021-01-01", periods=len(df), freq="H")
    return df

def generate_bull_flag():
    pole = np.linspace(90, 110, 40) + np.random.normal(0, 0.5, 40)
    flag = np.linspace(110, 108, 20) + np.random.normal(0, 0.3, 20)
    prices = np.concatenate([pole, flag])
    open_ = prices + np.random.normal(0, 0.2, len(prices))
    close = prices + np.random.normal(0, 0.2, len(prices))
    high = np.maximum(open_, close) + 0.3
    low = np.minimum(open_, close) - 0.3
    df = pd.DataFrame({"Open": open_, "High": high, "Low": low, "Close": close})
    df["Volume"] = np.random.randint(100, 1000, len(df))
    df.index = pd.date_range("2021-01-01", periods=len(df), freq="H")
    return df

def generate_head_and_shoulders():
    prices = np.array([100, 103, 100, 108, 100, 104, 100])
    prices = np.interp(np.linspace(0, len(prices) - 1, 70), np.arange(len(prices)), prices)
    prices += np.random.normal(0, 0.2, len(prices))
    open_ = prices + np.random.normal(0, 0.1, len(prices))
    close = prices + np.random.normal(0, 0.1, len(prices))
    high = np.maximum(open_, close) + 0.3
    low = np.minimum(open_, close) - 0.3
    df = pd.DataFrame({"Open": open_, "High": high, "Low": low, "Close": close})
    df["Volume"] = np.random.randint(100, 1000, len(df))
    df.index = pd.date_range("2021-01-01", periods=len(df), freq="H")
    return df

def generate_symmetrical_triangle():
    base = np.linspace(100, 100, 100)
    spread = np.linspace(5, 0.5, 100)
    upper = base + spread / 2 + np.random.normal(0, 0.2, 100)
    lower = base - spread / 2 + np.random.normal(0, 0.2, 100)
    close = base + np.random.normal(0, 0.3, 100)
    open_ = close + np.random.normal(0, 0.2, 100)
    df = pd.DataFrame({"Open": open_, "High": upper, "Low": lower, "Close": close})
    df["Volume"] = np.random.randint(100, 1000, len(df))
    df.index = pd.date_range("2021-01-01", periods=len(df), freq="H")
    return df

# === Save advanced pattern images ===
adv_patterns = {
    "cup_and_handle": generate_cup_and_handle,
    "falling_wedge": generate_falling_wedge,
    "bull_flag": generate_bull_flag,
    "head_and_shoulders": generate_head_and_shoulders,
    "symmetrical_triangle": generate_symmetrical_triangle,
}

for name, generator in adv_patterns.items():
    for i in range(100):
        df = generator()
        path = f"dataset/advanced/{name}_{i}.png"
        mpf.plot(df, type="candle", style="charles", savefig=path)

print("✅ Advanced pattern dataset generation complete.")


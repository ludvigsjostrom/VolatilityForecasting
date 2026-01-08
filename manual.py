# %% [markdown]
#  # Volatility Forecasting
#  
#  This notebook is a practical, end‑to‑end walkthrough of classical volatility forecasting using high‑frequency Binance BTCUSDT futures quote data. We start from raw quotes, construct volatility proxies (including realized variance from intraday returns), fit a ladder of econometric models, and compare them out of sample using appropriate loss functions.
#  
#  ## Roadmap
#  1. **Variance proxies**: squared returns, realized variance, and OHLC range estimators  
#  2. **Baselines**: rolling variance and EWMA  
#  3. **GARCH family**: GARCH(1,1), GJR‑GARCH, (optional) EGARCH  
#  4. **Realized‑vol models**: HAR‑RV on log realized variance  
#  5. **Stochastic volatility**: a state‑space approximation + (optional) Markov switching variance  
#  6. **Evaluation**: aligned targets, proper losses (QLIKE), and calibration diagnostics
#  
#  ## Data requirements
#  The notebook expects quote files on disk matching:
#  - `tardis_binance_btc/binance-futures_quotes_YYYY-MM-DD_BTCUSDT.csv`
#  - (optional) `tardis_binance_btc/binance-futures_trades_YYYY-MM-DD_BTCUSDT.csv`
#  - (optional) `tardis_binance_btc/binance-futures_book_snapshot_5_YYYY-MM-DD_BTCUSDT.csv`
#  
#  Two practical themes recur throughout:
#  - **Microstructure artifacts** (e.g., crossed/locked quotes) can create artificial midprice jumps if left untreated.
#  - **Very small return magnitudes** can make non‑linear estimation numerically fragile; we therefore work in **percent log‑return units** and scale variance proxies consistently.

# %% [markdown]
#  ## 0. Setup
#  
#  We begin by importing the scientific Python stack used throughout the notebook. Later sections rely on:
#  - `statsmodels` for regressions and regime‑switching models,
#  - `scipy.optimize` for likelihood‑based estimation of GARCH/SV parameters,
#  - `matplotlib` for sanity checks and diagnostics.

# %%
import warnings
warnings.filterwarnings("ignore")

from pathlib import Path
import glob

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt

import statsmodels.api as sm
from statsmodels.graphics.tsaplots import plot_acf

from scipy.optimize import minimize

print("pandas:", pd.__version__)
print("numpy :", np.__version__)
print("statsmodels:", sm.__version__)

# %% [markdown]
#  ### Configuration
#  
#  We work on two time grids:
#  - `BASE_FREQ`: **high‑frequency sampling grid** used to build realized variance (e.g., `"1s"`).
#  - `BAR_FREQ`: **modeling horizon** for returns and forecasts (e.g., `"5min"`).
#  
#  #### Why scale returns?
#  Bar log‑returns at short horizons are often very small $(10^{-4}$–(10^{-3}$). Many volatility models are estimated by non‑linear optimization, and tiny magnitudes can lead to ill‑conditioned problems. A standard remedy is to work in percentage points:
#  
#  $$
#  r^{(\%)}_t = 100 \cdot \log\left(\frac{P_t}{P_{t-1}}\right)
#  $$
#  
#  With `RETURN_SCALE = 100.0`, we compute `ret_pct = RETURN_SCALE * ret`.  
#  Any variance proxy built from returns is scaled by `RETURN_SCALE^2`, so everything is expressed in percent-squared units and remains directly comparable.

# %%
DATA_DIR = Path("tardis_binance_btc")
SYMBOL = "BTCUSDT"

BASE_FREQ = "1s"
BAR_FREQ  = "5min"

# Scale returns to improve numerical stability (percent log-returns)
RETURN_SCALE = 100.0

# CSV patterns (supports many days)
QUOTES_GLOB = str(DATA_DIR / f"binance-futures_quotes_*_{SYMBOL}.csv")
TRADES_GLOB = str(DATA_DIR / f"binance-futures_trades_*_{SYMBOL}.csv")
BOOK_GLOB   = str(DATA_DIR / f"binance-futures_book_snapshot_5_*_{SYMBOL}.csv")

quote_files = sorted(glob.glob(QUOTES_GLOB))
trade_files = sorted(glob.glob(TRADES_GLOB))
book_files  = sorted(glob.glob(BOOK_GLOB))

print("Quotes files:", len(quote_files))
print("Trades files:", len(trade_files))
print("Book snapshot files:", len(book_files))

# %% [markdown]
#  ## 0.1 Loaders and resampling helpers
#  
#  High‑frequency quote streams can contain artifacts (stale updates, out‑of‑order timestamps, or temporarily inconsistent bid/ask pairs). Before computing returns, we apply lightweight cleaning:
#  - remove non‑positive prices,
#  - optionally drop crossed or locked quotes where `ask <= bid`.
#  
#  **Why drop crossed/locked quotes?**  
#  A non‑positive spread is typically a synchronization or data issue rather than a tradable market state. Keeping such observations can create spurious midprice jumps and artificially inflate realized variance.
#  
#  We also adopt a clear timestamp convention: bars are **right‑labeled** (their timestamp is the bar close). This makes forecast alignment unambiguous when we store a forecast at time \(t\) meant to predict variance over \(t \rightarrow t+1\).

# %%
def _to_dt_us(x: pd.Series) -> pd.DatetimeIndex:
    """Convert integer microseconds since epoch to UTC datetime."""
    return pd.to_datetime(x.astype("int64"), unit="us", utc=True)

def load_quotes_mid(paths, chunksize=None, drop_crossed=True):
    """
    Load quote CSV(s) and return tick-level DataFrame indexed by dt with:
      - bid_price, ask_price, bid_amount, ask_amount
      - mid, spread

    Cleaning:
      - drop non-positive prices
      - optionally drop crossed/locked quotes (ask <= bid)
    """
    usecols = ["timestamp", "bid_price", "ask_price", "bid_amount", "ask_amount"]
    dtypes = {
        "timestamp": "int64",
        "bid_price": "float64",
        "ask_price": "float64",
        "bid_amount": "float64",
        "ask_amount": "float64",
    }
    frames = []
    for p in paths:
        if chunksize is None:
            df = pd.read_csv(p, usecols=usecols, dtype=dtypes)
            df["dt"] = _to_dt_us(df["timestamp"])
            df = df.set_index("dt").sort_index()
            frames.append(df)
        else:
            reader = pd.read_csv(p, usecols=usecols, dtype=dtypes, chunksize=chunksize)
            for chunk in reader:
                chunk["dt"] = _to_dt_us(chunk["timestamp"])
                chunk = chunk.set_index("dt")
                frames.append(chunk)

    df = pd.concat(frames).sort_index()
    df = df[~df.index.duplicated(keep="last")]

    # Basic cleaning
    df = df[(df["bid_price"] > 0) & (df["ask_price"] > 0)].copy()
    df["spread"] = df["ask_price"] - df["bid_price"]
    crossed = (df["spread"] <= 0).sum()
    if drop_crossed:
        df = df[df["spread"] > 0].copy()

    df["mid"] = 0.5 * (df["bid_price"] + df["ask_price"])
    df["imbalance_L1"] = (df["bid_amount"] - df["ask_amount"]) / (df["bid_amount"] + df["ask_amount"] + 1e-12)

    info = {
        "rows_after_concat": int(sum(1 for _ in frames)) if chunksize is not None else None,
        "crossed_or_locked_quotes": int(crossed),
        "final_rows": int(len(df)),
    }
    return df[["bid_price","ask_price","bid_amount","ask_amount","mid","spread","imbalance_L1"]], info

def make_time_bars_from_mid(mid_df: pd.DataFrame, freq: str):
    """Build OHLC bars from midprice (right-labeled, right-closed).
    
    Convention: each bar's timestamp is the **bar close** (right edge).
    This avoids ambiguity when aligning t→t+1 forecasts.
    """
    mid = mid_df["mid"].copy()
    ohlc = mid.resample(freq, label="right", closed="right").ohlc()
    spread = mid_df["spread"].resample(freq, label="right", closed="right").last().rename("spread")
    imb = mid_df["imbalance_L1"].resample(freq, label="right", closed="right").last().rename("imbalance_L1")
    out = ohlc.join([spread, imb], how="left")
    return out

def compute_log_returns_from_close(close: pd.Series) -> pd.Series:
    close = close.replace([0, np.inf, -np.inf], np.nan).dropna()
    return np.log(close).diff().dropna()

def realized_variance_from_base_returns(ret_base: pd.Series, bar_freq: str) -> pd.Series:
    """RV per bar = sum of squared base-frequency returns inside each bar."""
    return ret_base.pow(2).resample(bar_freq, label="right", closed="right").sum().dropna()

# %% [markdown]
#  ## 0.2 Load data and build the two grids
#  
#  We now:
#  1. Load tick‑level quotes and compute the midprice.
#  2. Resample midprice to the **base grid** (`BASE_FREQ`) to compute intrabar returns for realized variance.
#  3. Aggregate midprice to **OHLC bars** at the modeling horizon (`BAR_FREQ`) to compute bar returns.
# 
#  We maintain two aligned objects:
#  
#  - **Base grid** (high frequency): used to build realized variance within each `BAR_FREQ` interval.
#  - **Bar grid** (modeling grid): where we fit models and produce one‑step‑ahead forecasts.
#  
#  On the bar grid we construct:
#  - `ret`: bar log return \(r_t\),
#  - `ret_pct`: scaled return \(r^{(\%)}_t = \texttt{RETURN\_SCALE}\cdot r_t\),
#  - `RV`: realized variance per bar, as the sum of squared base‑grid returns inside the bar,
#  - `RV_pct2`: realized variance in percent‑squared units, \(\texttt{RETURN\_SCALE}^2 \cdot RV_t\).
#  
#  **Forecasting alignment note:** a forecast stored at index \(t\) is intended to be available at the **close of bar \(t\)** and to target the next bar’s realized variance \(RV_{\%^2, t+1}\).
# 

# %%
mid_tick, info = load_quotes_mid(quote_files, chunksize=None, drop_crossed=True)
print("Loaded tick rows:", len(mid_tick))
print("Crossed/locked quotes found (dropped):", info["crossed_or_locked_quotes"])
print(mid_tick.head())

# Base grid mid and returns (right-labeled timestamps = base-grid "closes")
mid_base = mid_tick["mid"].resample(BASE_FREQ, label="right", closed="right").last().ffill()
ret_base = compute_log_returns_from_close(mid_base)

# Bar grid OHLC and returns
bars = make_time_bars_from_mid(mid_tick, BAR_FREQ).dropna(subset=["close"])
bars["ret"] = compute_log_returns_from_close(bars["close"])
bars["ret_pct"] = RETURN_SCALE * bars["ret"]

# Realized variance per bar from base returns
rv_bar = realized_variance_from_base_returns(ret_base, BAR_FREQ)
bars = bars.join(rv_bar.rename("RV"), how="inner")
bars["RV_pct2"] = (RETURN_SCALE**2) * bars["RV"]

# Drop NA returns
df = bars.dropna(subset=["ret_pct","RV_pct2"]).copy()

print(df[["open","high","low","close","spread","ret_pct","RV_pct2"]].head())
print("Bars:", len(df), "| span:", df.index.min(), "→", df.index.max())


# %% [markdown]
#  ## 0.3 Quick exploratory plots
#  
#  Before modeling, we run a few fast diagnostics:
#  - **Price level**: confirms resampling/aggregation behaves as expected.
#  - **Bid‑ask spread**: spikes can coincide with stress periods or data issues.
#  - **Absolute returns**: highlights volatility bursts and quiet regimes.
#  - **ACF of squared returns**: checks for **volatility clustering** (slow decay in autocorrelation of \(r_t^2\)).
#  
#  These plots help catch problems early (e.g., missing-data flats, abnormal spreads) and build intuition for the horizon being forecast.

# %%
fig, ax = plt.subplots(figsize=(12, 4))
ax.plot(df.index, df["close"])
ax.set_title(f"{SYMBOL} mid (bar close) at {BAR_FREQ}")
ax.set_ylabel("Price")
plt.show()

fig, ax = plt.subplots(figsize=(12, 3))
ax.plot(df.index, df["spread"])
ax.set_title("Bid-ask spread (last in bar)")
ax.set_ylabel("Spread")
plt.show()

fig, ax = plt.subplots(figsize=(12, 3))
ax.plot(df.index, df["ret_pct"].abs())
ax.set_title(f"Absolute returns (bar, in % log-return units; scale={RETURN_SCALE})")
ax.set_ylabel("|return| (%)")
plt.show()

fig = plt.figure(figsize=(10, 4))
plot_acf((df["ret_pct"]**2), lags=50, ax=plt.gca())
plt.title("ACF of squared returns (volatility clustering)")
plt.show()

# Spread sanity check distribution
fig, ax = plt.subplots(figsize=(6, 3))
ax.hist(df["spread"].dropna(), bins=60)
ax.set_title("Spread distribution (bar last)")
plt.show()

# %% [markdown]
# 
#  ## 1. Start with proxies
#  
#  Volatility is latent: we do not observe the “true” conditional variance directly. In practice, we evaluate variance forecasts against an observable **proxy** constructed from price data. The proxy’s noise level matters—especially at short horizons—because noisy targets can mask differences between models.
# 
# 
#  We will compare models using bar‑horizon proxies:
#  
#  - **Squared bar return** \( (r^{(\%)}_t)^2 \)  
#    Always available, but extremely noisy as an estimator of integrated variance over a short bar.
#  
#  - **Realized variance (RV)** from higher‑frequency returns inside the bar  
#    Often a far better proxy for bar‑level integrated variance, provided the base sampling frequency is chosen sensibly.
#  
#  Because we work in percent units, all variances are in **percent-squared**:
#  \[
#  r^2_{\%,t} = \left(r^{(\%)}_t\right)^2, \qquad
#  RV_{\%^2,t} = \texttt{RETURN\_SCALE}^2 \cdot \sum_{i \in \text{bar } t} r_{i}^2.
#  \]
#  
#  In later sections, the **forecast target** is the *next-bar* realized variance \(RV_{\%^2, t+1}\).

# %%
df["r2_pct2"] = df["ret_pct"]**2

fig, ax = plt.subplots(figsize=(12, 4))
ax.plot(df.index, df["r2_pct2"], label="(bar return)^2 (pct^2)", alpha=0.7)
ax.plot(df.index, df["RV_pct2"], label=f"RV from {BASE_FREQ} returns (pct^2)", alpha=0.7)
ax.set_yscale("log")
ax.set_title("Variance proxies on the same horizon (log scale)")
ax.set_ylabel("Variance (pct^2)")
ax.legend()
plt.show()

print("Corr(r^2, RV):", df["r2_pct2"].corr(df["RV_pct2"]))

# %% [markdown]
#  ### 1.1 Range-based variance estimators (OHLC)
#  
#  When you have OHLC bars (or you want a complementary proxy), **range-based estimators** use intrabar high/low (and sometimes open/close) to estimate variance. Under idealized assumptions they can be more informative than \(r_t^2\), though at very short horizons they may also be sensitive to microstructure noise.
# 
# 
#  For each `BAR_FREQ` interval, we compute three widely used estimators:
#  
#  - **Parkinson (1980)**: uses the high/low range; efficient under driftless Brownian motion.
#  - **Garman–Klass (1980)**: combines high/low and open/close; often more efficient when assumptions hold.
#  - **Rogers–Satchell (1991)**: allows for non‑zero drift within the bar (useful when the price trends during the interval).
#  
#  These formulas produce variance in **log‑return units**; we multiply by `RETURN_SCALE^2` to express them in the same **percent-squared** units as `RV_pct2`.
# 

# %%
def parkinson_var(ohlc: pd.DataFrame) -> pd.Series:
    hl = np.log(ohlc["high"] / ohlc["low"])
    return (hl**2) / (4.0 * np.log(2.0))

def garman_klass_var(ohlc: pd.DataFrame) -> pd.Series:
    hl = np.log(ohlc["high"] / ohlc["low"])
    co = np.log(ohlc["close"] / ohlc["open"])
    return 0.5 * (hl**2) - (2.0*np.log(2.0) - 1.0) * (co**2)

def rogers_satchell_var(ohlc: pd.DataFrame) -> pd.Series:
    ho = np.log(ohlc["high"] / ohlc["open"])
    lo = np.log(ohlc["low"] / ohlc["open"])
    co = np.log(ohlc["close"] / ohlc["open"])
    return ho*(ho - co) + lo*(lo - co)

df["parkinson_pct2"] = (RETURN_SCALE**2) * parkinson_var(df)
df["gk_pct2"] = (RETURN_SCALE**2) * garman_klass_var(df)
df["rs_pct2"] = (RETURN_SCALE**2) * rogers_satchell_var(df)

fig, ax = plt.subplots(figsize=(12, 4))
ax.plot(df.index, df["RV_pct2"], label="RV target", alpha=0.7)
ax.plot(df.index, df["parkinson_pct2"], label="Parkinson", alpha=0.7)
ax.plot(df.index, df["gk_pct2"], label="Garman–Klass", alpha=0.7)
ax.plot(df.index, df["rs_pct2"], label="Rogers–Satchell", alpha=0.7)
ax.set_yscale("log")
ax.set_title("Range-based variance estimators vs realized variance (log scale)")
ax.set_ylabel("Variance (pct^2)")
ax.legend()
plt.show()

print(df[["RV_pct2","parkinson_pct2","gk_pct2","rs_pct2"]].describe(percentiles=[0.5, 0.9, 0.99]))

# %% [markdown]
#  ### 1.2 Sampling frequency sensitivity (a quick “signature” check)
#  
#  Realized variance depends on the sampling frequency used to build intraday returns. Sampling *too slowly* misses intrabar variation; sampling *too fast* can inflate RV due to **microstructure noise** (bid/ask bounce, price discreteness, asynchronous updates).
#  
#  As a quick diagnostic, we compute RV on the same `BAR_FREQ` horizon using several base frequencies and compare them:
#  - If very high‑frequency RV is systematically larger and less correlated with coarser RV, microstructure noise may be dominating.
#  - A base frequency with stable behavior (often around 1–10 seconds for liquid markets) is a practical compromise.
# 

# %%
def rv_at_basefreq(mid_tick: pd.DataFrame, base_freq: str, bar_freq: str) -> pd.Series:
    mid_base = mid_tick["mid"].resample(base_freq, label="right", closed="right").last().ffill()
    ret_base = compute_log_returns_from_close(mid_base)
    return realized_variance_from_base_returns(ret_base, bar_freq)

base_freqs = ["250ms", "500ms", "1s", "2s", "5s", "10s"]
rv_compare = pd.DataFrame({bf: rv_at_basefreq(mid_tick, bf, BAR_FREQ) for bf in base_freqs}).dropna()
rv_compare_pct2 = (RETURN_SCALE**2) * rv_compare

fig, ax = plt.subplots(figsize=(12, 4))
for bf in base_freqs:
    ax.plot(rv_compare_pct2.index, rv_compare_pct2[bf], label=bf, alpha=0.7)
ax.set_yscale("log")
ax.set_title(f"Realized variance per {BAR_FREQ} using different base sampling frequencies (log scale)")
ax.set_ylabel("RV (pct^2)")
ax.legend(ncol=3)
plt.show()

if "1s" in rv_compare.columns:
    corr_to_1s = rv_compare.corr()["1s"].sort_values(ascending=False)
    print("Correlation to 1s RV:")
    print(corr_to_1s)

# %% [markdown]
#  ## 2. Baseline models: rolling variance and EWMA
#  
#  Before fitting parametric volatility models, we establish strong baselines. A baseline that is hard to beat is valuable: if a complex model cannot improve out of sample, it is usually not worth the added complexity.
# 
#  We forecast **next‑bar variance** at each bar close.
#  
#  - Target (observed after the next bar completes):
#  \[
#  y_t = RV_{\%^2,\,t+1}
#  \]
#  - Forecast produced using information up to bar \(t\):
#  \[
#  \hat h_t = \hat h_{t+1|t}
#  \]
#  
#  Two baselines:
#  - **Rolling variance**: trailing mean of squared returns.
#  - **EWMA**: exponentially weighted moving average that reacts faster to changes in volatility.
#  
#  Both produce forecasts in percent‑squared units, directly comparable to the realized variance target.
# 
#  ### 2.1 Train/test split
#  
#  Time‑series evaluation must respect chronology. We split the sample into an initial **training** period and a later **test** period.
#  
#  A subtle but important detail is target alignment. Because our label is **next‑bar RV**, we must avoid leaking information across the split boundary. The procedure is:
#  1. Split the unshifted series into train and test.
#  2. Create `target_RV_next = RV_pct2.shift(-1)` **within each split**.
#  
#  This ensures the last training observation does **not** use the first test RV as its label.
# 

# %%
df_model = df[["ret_pct","RV_pct2","r2_pct2","parkinson_pct2","gk_pct2","rs_pct2","spread","imbalance_L1"]].copy()

split_frac = 0.7
split_idx = int(len(df_model) * split_frac)
train_raw = df_model.iloc[:split_idx].copy()
test_raw  = df_model.iloc[split_idx:].copy()

# Then define next-bar targets WITHIN each split (no cross-contamination)
train_raw["target_RV_next"] = train_raw["RV_pct2"].shift(-1)
test_raw["target_RV_next"]  = test_raw["RV_pct2"].shift(-1)

train = train_raw.dropna()
test  = test_raw.dropna()

# Also keep a full df_model for rolling/EWMA features (computed on whole series, but evaluated separately)
df_model["target_RV_next"] = df_model["RV_pct2"].shift(-1)

print("Train:", train.index.min(), "→", train.index.max(), "| n=", len(train))
print("Test :", test.index.min(), "→", test.index.max(),  "| n=", len(test))

# %% [markdown]
#  ### 2.2 Rolling variance forecasts
#  
#  The rolling estimator uses the recent history of squared returns:
#  \[
#  \hat h_{t+1|t} = \frac{1}{W}\sum_{j=0}^{W-1} (r^{(\%)}_{t-j})^2
#  \]
#  where \(W\) is the window length in bars.
#  
#  Rolling variance is transparent and robust, but it reacts slowly to abrupt volatility shifts. We try several window sizes to illustrate the responsiveness vs. smoothness trade‑off.
# 

# %%
def rolling_var_forecast(ret_pct: pd.Series, window: int) -> pd.Series:
    return ret_pct.pow(2).rolling(window=window).mean()

windows = [12, 36, 72]
for w in windows:
    # Forecast stored at index t represents a t→t+1 forecast (available at bar close t)
    df_model[f"roll_{w}"] = rolling_var_forecast(df_model["ret_pct"], w)

fig, ax = plt.subplots(figsize=(12, 4))
ax.plot(test.index, test["target_RV_next"], label="Target RV next (pct^2)", alpha=0.7)
for w in windows:
    ax.plot(test.index, df_model.loc[test.index, f"roll_{w}"], label=f"Rolling r^2, w={w}", alpha=0.7)
ax.set_yscale("log")
ax.set_title("Rolling variance forecasts vs next-bar realized variance (log scale)")
ax.set_ylabel("Variance (pct^2)")
ax.legend()
plt.show()

# %% [markdown]
# 
#  ### 2.3 EWMA variance forecasts
#  
#  EWMA updates variance recursively:
#  \[
#  h_t = \lambda h_{t-1} + (1-\lambda)(r^{(\%)}_{t-1})^2
#  \]
#  After observing \(r^{(\%)}_t\), the **one‑step‑ahead** forecast is:
#  \[
#  \hat h_{t+1|t} = \lambda h_t + (1-\lambda)(r^{(\%)}_t)^2.
#  \]
#  
#  The decay parameter \(\lambda \in (0,1)\) controls responsiveness:
#  - smaller \(\lambda\): reacts quickly but is noisier,
#  - larger \(\lambda\): smoother but slower to adapt.
#  
#  We also compute the **half‑life** implied by \(\lambda\) to translate it into an intuitive “memory length” in bars.
# 

# %%
def ewma_variance(ret_pct: pd.Series, lam: float, init_var=None) -> pd.Series:
    r2 = ret_pct.pow(2).values
    h = np.empty_like(r2)
    if init_var is None:
        init_var = np.nanmean(r2[:100]) if len(r2) > 100 else np.nanmean(r2)
    h[0] = max(init_var, 1e-12)
    for t in range(1, len(r2)):
        h[t] = lam * h[t-1] + (1.0 - lam) * r2[t-1]
    return pd.Series(h, index=ret_pct.index)

def ewma_variance_next(ret_pct: pd.Series, lam: float, init_var=None) -> pd.Series:
    """EWMA next-step forecast: h_{t+1|t} at index t.
    
    After observing r_t, the forecast for variance at t+1 is:
    h_{t+1|t} = λ*h_t + (1-λ)*r²_t
    
    where h_t = λ*h_{t-1} + (1-λ)*r²_{t-1} is the conditional variance for r_t given info up to t-1.
    """
    h_t = ewma_variance(ret_pct, lam, init_var=init_var)
    r2_t = ret_pct.pow(2)
    # One-step-ahead forecast available at bar close t
    return lam * h_t + (1.0 - lam) * r2_t

lams = [0.94, 0.97, 0.99]
for lam in lams:
    df_model[f"ewma_{lam}_fcst"] = ewma_variance_next(df_model["ret_pct"], lam)

fig, ax = plt.subplots(figsize=(12, 4))
ax.plot(test.index, test["target_RV_next"], label="Target RV next", alpha=0.7)
for lam in lams:
    ax.plot(test.index, df_model.loc[test.index, f"ewma_{lam}_fcst"], label=f"EWMA λ={lam}", alpha=0.7)
ax.set_yscale("log")
ax.set_title("EWMA variance forecasts vs next-bar RV (log scale)")
ax.set_ylabel("Variance (pct^2)")
ax.legend()
plt.show()

for lam in lams:
    half_life = np.log(0.5) / np.log(lam)
    print(f"λ={lam}: half-life ≈ {half_life:.2f} bars ({half_life * pd.Timedelta(BAR_FREQ)})")

# %% [markdown]
#  ### 2.4 Tune EWMA \( \lambda \) by QLIKE
#  
#  To select \(\lambda\) in a way that matches volatility‑forecasting objectives, we tune it on the training set using **QLIKE**, a common loss for variance forecasts evaluated against realized variance:
#  \[
#  \text{QLIKE}(y,h) = \frac{y}{h} - \log\left(\frac{y}{h}\right) - 1.
#  \]
#  
#  QLIKE is widely used because it behaves well with noisy realized measures and is closely connected to likelihood‑based evaluation. We perform a simple grid search and carry the best \(\lambda\) forward to the test comparison.
# 

# %%
def qlike_loss(y_true: pd.Series, y_pred: pd.Series, eps=1e-12) -> float:
    y = np.maximum(y_true.values, eps)
    h = np.maximum(y_pred.values, eps)
    ratio = y / h
    return float(np.mean(ratio - np.log(ratio) - 1.0))

def ewma_qlike_objective(lam, ret_train, target_train):
    lam = float(lam)
    if not (0.0 < lam < 1.0):
        return 1e9
    h = ewma_variance_next(ret_train, lam)
    aligned = pd.concat([target_train, h], axis=1).dropna()
    return qlike_loss(aligned.iloc[:,0], aligned.iloc[:,1])

lam_grid = np.linspace(0.70, 0.999, 100)
losses = [ewma_qlike_objective(lam, train["ret_pct"], train["target_RV_next"]) for lam in lam_grid]
best_lam = float(lam_grid[int(np.argmin(losses))])
print("Best λ on grid:", best_lam)

fig, ax = plt.subplots(figsize=(10, 3))
ax.plot(lam_grid, losses)
ax.set_title("EWMA λ tuning via training QLIKE")
ax.set_xlabel("λ")
ax.set_ylabel("Mean QLIKE")
plt.show()

df_model["ewma_tuned_fcst"] = ewma_variance_next(df_model["ret_pct"], best_lam)

# %% [markdown]
#  ## 3. GARCH tier
#  
#  Baselines treat volatility as a smoothed function of past squared returns. **GARCH-family** models specify an explicit conditional variance process and estimate parameters via (quasi) maximum likelihood.
#  
#  We focus on widely used one‑lag specifications:
#  - **GARCH(1,1)** for symmetric volatility dynamics,
#  - **GJR‑GARCH(1,1)** to capture leverage/asymmetry,
#  - **EGARCH(1,1)** (optional) in log‑variance form.
# 
# 
#  We estimate these models using a **Gaussian quasi log-likelihood**. Even when returns are heavy‑tailed, Gaussian QML often produces useful variance forecasts.
#  
#  A practical challenge is that GARCH parameters must satisfy positivity and stationarity constraints. Rather than relying on fragile constrained optimizers, we use **smooth reparameterizations** that enforce admissibility *by construction*. This keeps variance positive during optimization and improves numerical stability.
#  
#  ### 3.1 Gaussian log-likelihood and variance recursions
#  The next cell defines the Gaussian negative log-likelihood and the recursive filters that map returns \(r_t\) into a conditional variance path \(h_t\).
# 

# %%
def gaussian_nll(r: np.ndarray, h: np.ndarray):
    h = np.maximum(h, 1e-12)
    return 0.5 * np.sum(np.log(2*np.pi) + np.log(h) + (r**2)/h)

def garch11_filter(r: np.ndarray, omega: float, alpha: float, beta: float, h0=None):
    T = len(r)
    h = np.empty(T)
    if h0 is None:
        h0 = np.var(r)
    h[0] = max(h0, 1e-12)
    for t in range(1, T):
        h[t] = omega + alpha * r[t-1]**2 + beta * h[t-1]
        h[t] = max(h[t], 1e-12)
    return h

def gjr_filter(r: np.ndarray, omega: float, alpha: float, gamma: float, beta: float, h0=None):
    T = len(r)
    h = np.empty(T)
    if h0 is None:
        h0 = np.var(r)
    h[0] = max(h0, 1e-12)
    for t in range(1, T):
        ind = 1.0 if r[t-1] < 0 else 0.0
        h[t] = omega + (alpha + gamma*ind) * r[t-1]**2 + beta * h[t-1]
        h[t] = max(h[t], 1e-12)
    return h

# %% [markdown]
#  ### 3.2 Fit GARCH(1,1) with a stationary-by-construction parameterization
#  
#  The standard GARCH(1,1) model is:
#  \[
#  h_t = \omega + \alpha r_{t-1}^2 + \beta h_{t-1}, \qquad \omega>0,\ \alpha\ge 0,\ \beta\ge 0,\ \alpha+\beta<1.
#  \]
#  
#  We enforce these constraints automatically through a smooth mapping from unconstrained parameters. This avoids invalid variance values during optimization and guarantees a stationary fitted process.
# 
#  %% [markdown]
#  **Reparameterization**
#  
#  We parameterize in terms of:
#  - Unconditional variance: \(\bar h = \exp(\theta_0)\),
#  - Persistence: \(p = \sigma(\theta_1)(1-\varepsilon)\),
#  - Share: \(s = \sigma(\theta_2)\),
#  
#  and map to:
#  \[
#  \alpha = sp,\qquad \beta=(1-s)p,\qquad \omega=\bar h(1-p).
#  \]
#  
#  This guarantees \(\omega>0\), \(\alpha\ge 0\), \(\beta\ge 0\), and \(\alpha+\beta=p<1\) by construction, while allowing unconstrained optimization over \(\theta\).
# 
# 

# %%
def sigmoid(x):
    return 1.0 / (1.0 + np.exp(-x))

def fit_garch11_reparam(ret_pct: pd.Series, eps=1e-6):
    x = ret_pct.values
    var = np.var(x)

    # theta = [log_hbar, logit_p, logit_s]
    theta0 = np.array([np.log(max(var,1e-8)), 2.0, 0.0])  # p ~ sigmoid(2)=0.88, s=0.5

    def unpack(theta):
        log_hbar, a, b = theta
        hbar = np.exp(log_hbar)
        p = sigmoid(a) * (1.0 - eps)
        s = sigmoid(b)
        alpha = s * p
        beta = (1.0 - s) * p
        omega = hbar * (1.0 - p)
        return omega, alpha, beta, hbar, p, s

    def obj(theta):
        omega, alpha, beta, *_ = unpack(theta)
        h = garch11_filter(x, omega, alpha, beta)
        return gaussian_nll(x, h)

    res = minimize(obj, theta0, method="L-BFGS-B")
    omega, alpha, beta, hbar, p, s = unpack(res.x)
    h = garch11_filter(x, omega, alpha, beta)
    return {
        "omega": omega, "alpha": alpha, "beta": beta,
        "hbar": hbar, "p": p, "share_alpha": s,
        "success": res.success, "message": res.message, "nll": res.fun,
        "h": pd.Series(h, index=ret_pct.index)
    }

# Fit on train_raw (not train) to avoid the one-bar gap at the boundary.
# GARCH uses only returns, so this does not reintroduce label leakage.
garch_fit = fit_garch11_reparam(train_raw["ret_pct"])
print(garch_fit)

fig, ax = plt.subplots(figsize=(12, 4))
ax.plot(train_raw.index, garch_fit["h"], label="GARCH(1,1) h_t (train_raw)")
ax.set_yscale("log")
ax.set_title("GARCH(1,1) fitted conditional variance (log scale)")
ax.set_ylabel("Variance (pct^2)")
ax.legend()
plt.show()

# %% [markdown]
#  ### 3.3 Fit GJR‑GARCH(1,1) with asymmetry
#  
#  Many markets exhibit asymmetric volatility response: negative returns tend to increase future volatility more than positive returns of the same size. GJR‑GARCH captures this via an indicator term:
#  \[
#  h_t = \omega + (\alpha + \gamma\,\mathbb{1}\{r_{t-1}<0\})r_{t-1}^2 + \beta h_{t-1}.
#  \]
# 
#  A common stationarity condition is:
#  \[
#  \alpha + \beta + \tfrac{1}{2}\gamma < 1.
#  \]
#  
#  We enforce admissibility with a reparameterization:
#  - \(\bar h = \exp(\theta_0)\),
#  - \(p = \sigma(\theta_1)(1-\varepsilon)\),
#  - non‑negative weights \(w=(w_\alpha,w_\beta,w_\gamma)\) from a softmax so \(w_\alpha+w_\beta+w_\gamma=1\).
#  
#  Mapping:
#  \[
#  \alpha=w_\alpha p,\quad \beta=w_\beta p,\quad \gamma=2w_\gamma p,\quad \omega=\bar h(1-p),
#  \]
#  implying \(\alpha+\beta+\gamma/2=p<1\) automatically.
# 

# %%
def softmax(v):
    v = np.asarray(v)
    v = v - np.max(v)
    e = np.exp(v)
    return e / np.sum(e)

def fit_gjr_reparam(ret_pct: pd.Series, eps=1e-6):
    x = ret_pct.values
    var = np.var(x)

    theta0 = np.array([np.log(max(var,1e-8)), 2.0, 0.0, 0.0, 0.0])  # log_hbar, logit_p, logits weights

    def unpack(theta):
        log_hbar, a, u1, u2, u3 = theta
        hbar = np.exp(log_hbar)
        p = sigmoid(a) * (1.0 - eps)
        w = softmax([u1,u2,u3])
        alpha = w[0] * p
        beta  = w[1] * p
        gamma = 2.0 * w[2] * p
        omega = hbar * (1.0 - p)
        return omega, alpha, gamma, beta, hbar, p, w

    def obj(theta):
        omega, alpha, gamma, beta, *_ = unpack(theta)
        h = gjr_filter(x, omega, alpha, gamma, beta)
        return gaussian_nll(x, h)

    res = minimize(obj, theta0, method="L-BFGS-B")
    omega, alpha, gamma, beta, hbar, p, w = unpack(res.x)
    h = gjr_filter(x, omega, alpha, gamma, beta)
    return {
        "omega": omega, "alpha": alpha, "gamma": gamma, "beta": beta,
        "hbar": hbar, "p": p, "weights": w,
        "success": res.success, "message": res.message, "nll": res.fun,
        "h": pd.Series(h, index=ret_pct.index)
    }

# Fit on train_raw to match GARCH (no gap at boundary)
gjr_fit = fit_gjr_reparam(train_raw["ret_pct"])
print(gjr_fit)

fig, ax = plt.subplots(figsize=(12, 4))
ax.plot(train_raw.index, garch_fit["h"], label="GARCH(1,1)", alpha=0.7)
ax.plot(train_raw.index, gjr_fit["h"], label="GJR-GARCH(1,1)", alpha=0.7)
ax.set_yscale("log")
ax.set_title("Conditional variance: symmetric vs asymmetric GARCH (log scale)")
ax.set_ylabel("Variance (pct^2)")
ax.legend()
plt.show()

# %% [markdown]
#  ### 3.4 EGARCH(1,1) (optional): multi-start estimation and basic sanity checks
#  
#  EGARCH models log variance:
#  \[
#  \log h_t = \omega + \beta \log h_{t-1} + \alpha\left(|z_{t-1}|-\mathbb{E}|z|\right) + \gamma z_{t-1},\quad z_{t-1}=\frac{r_{t-1}}{\sqrt{h_{t-1}}}.
#  \]
#  
#  Likelihood surfaces for EGARCH can be sensitive to starting values, especially on short samples. To improve reliability we:
#  - fit on scaled returns,
#  - try a small set of sensible initializations (multi‑start),
#  - discard fits whose implied average variance is wildly inconsistent with the data.
# 
#  If no stable EGARCH solution is found on the current sample, that is not unusual. EGARCH often benefits from longer histories and/or more robust error distributions. In many applications, GJR‑GARCH and realized‑volatility models offer a better complexity‑to‑benefit trade‑off.
# 

# %%
def egarch_filter(r: np.ndarray, omega: float, alpha: float, gamma: float, beta: float, h0=None):
    T = len(r)
    logh = np.empty(T)
    if h0 is None:
        h0 = np.var(r)
    logh[0] = np.log(max(h0, 1e-12))
    Ez = np.sqrt(2.0/np.pi)
    for t in range(1, T):
        h_prev = np.exp(logh[t-1])
        z_prev = r[t-1] / np.sqrt(max(h_prev, 1e-12))
        logh[t] = omega + beta*logh[t-1] + alpha*(np.abs(z_prev) - Ez) + gamma*z_prev
    return np.exp(logh)

def fit_egarch_multistart(ret_pct: pd.Series, starts=None):
    x = ret_pct.values
    var = np.var(x)
    if starts is None:
        # start omega roughly consistent with unconditional log-variance
        # For beta ~0.95, omega approx (1-beta)*log(var)
        starts = []
        for beta in [0.90, 0.95, 0.98]:
            omega = (1-beta) * np.log(max(var,1e-8))
            for alpha in [0.05, 0.10, 0.20]:
                for gamma in [-0.20, -0.10, 0.0]:
                    starts.append(np.array([omega, alpha, gamma, np.arctanh(np.clip(beta, -0.999, 0.999))]))
    bounds = [(-20, 20), (-2, 2), (-2, 2), (-5, 5)]  # last is raw for tanh beta

    def obj(p):
        omega, alpha, gamma, b_raw = p
        beta = np.tanh(b_raw)
        h = egarch_filter(x, omega, alpha, gamma, beta)
        return gaussian_nll(x, h)

    best = None
    for x0 in starts:
        res = minimize(obj, x0, method="L-BFGS-B", bounds=bounds, options={"maxiter": 500})
        if not res.success:
            continue
        omega, alpha, gamma, b_raw = res.x
        beta = np.tanh(b_raw)
        h = egarch_filter(x, omega, alpha, gamma, beta)
        # sanity: mean h should be in the ballpark of return variance
        if np.mean(h) > 100 * var:   # too large
            continue
        if np.mean(h) < 0.01 * var:  # too small
            continue
        cand = {"omega": omega, "alpha": alpha, "gamma": gamma, "beta": beta,
                "success": True, "message": res.message, "nll": res.fun,
                "h": pd.Series(h, index=ret_pct.index)}
        if best is None or cand["nll"] < best["nll"]:
            best = cand
    return best

# Fit on train_raw to match GARCH/GJR (no gap at boundary)
egarch_fit = fit_egarch_multistart(train_raw["ret_pct"])
if egarch_fit is None:
    print("EGARCH: no stable fit found on this sample.")
else:
    print("EGARCH fit:", {k: egarch_fit[k] for k in ["omega","alpha","gamma","beta","nll"]})
    fig, ax = plt.subplots(figsize=(12, 4))
    ax.plot(train_raw.index, egarch_fit["h"], label="EGARCH(1,1)")
    ax.set_yscale("log")
    ax.set_title("EGARCH conditional variance (log scale)")
    ax.set_ylabel("Variance (pct^2)")
    ax.legend()
    plt.show()

# %% [markdown]
#  ### 3.5 Out-of-sample one-step-ahead variance forecasts
#  
#  We now generate **one‑step‑ahead** variance forecasts on the test set for:
#  - rolling variance and tuned EWMA baselines,
#  - GARCH(1,1) and GJR‑GARCH (and EGARCH if available).
#  
#  Two details matter for clean out‑of‑sample forecasting:
#  1. **Forecast alignment:** each value stored at time \(t\) is \(h_{t+1|t}\), available at the close of bar \(t\).
#  2. **State initialization:** recursive models need a starting variance at the first test timestamp. We initialize from the end of the training filter and advance one step using the last training return.
# 

# %%
def forecast_path_garch11(ret_pct: pd.Series, fit: dict, h0: float):
    x = ret_pct.values
    omega, alpha, beta = fit["omega"], fit["alpha"], fit["beta"]
    h = np.empty(len(x))
    h[0] = max(h0, 1e-12)
    for t in range(1, len(x)):
        h[t] = omega + alpha*x[t-1]**2 + beta*h[t-1]
        h[t] = max(h[t], 1e-12)
    return pd.Series(h, index=ret_pct.index)

def forecast_path_gjr(ret_pct: pd.Series, fit: dict, h0: float):
    x = ret_pct.values
    omega, alpha, gamma, beta = fit["omega"], fit["alpha"], fit["gamma"], fit["beta"]
    h = np.empty(len(x))
    h[0] = max(h0, 1e-12)
    for t in range(1, len(x)):
        ind = 1.0 if x[t-1] < 0 else 0.0
        h[t] = omega + (alpha + gamma*ind)*x[t-1]**2 + beta*h[t-1]
        h[t] = max(h[t], 1e-12)
    return pd.Series(h, index=ret_pct.index)

def forecast_path_egarch(ret_pct: pd.Series, fit: dict, h0: float):
    x = ret_pct.values
    omega, alpha, gamma, beta = fit["omega"], fit["alpha"], fit["gamma"], fit["beta"]
    h = egarch_filter(x, omega, alpha, gamma, beta, h0=h0)
    return pd.Series(h, index=ret_pct.index)

# State starts: compute h0_test = h_{t0} where t0 is the first test timestamp
# h_{t0} = ω + α*r²_{T_train} + β*h_{T_train}  (one step ahead from last train_raw)
# Using train_raw ensures we include the boundary bar (last bar before test)
r_last_train = float(train_raw["ret_pct"].iloc[-1])

# GARCH(1,1) initialization for test
h_last_garch = float(garch_fit["h"].iloc[-1])
h0_test_garch = (
    garch_fit["omega"]
    + garch_fit["alpha"] * (r_last_train ** 2)
    + garch_fit["beta"] * h_last_garch
)

# GJR-GARCH initialization for test
h_last_gjr = float(gjr_fit["h"].iloc[-1])
ind_last = 1.0 if r_last_train < 0 else 0.0
h0_test_gjr = (
    gjr_fit["omega"]
    + (gjr_fit["alpha"] + gjr_fit["gamma"] * ind_last) * (r_last_train ** 2)
    + gjr_fit["beta"] * h_last_gjr
)

h_test_garch = forecast_path_garch11(test["ret_pct"], garch_fit, h0_test_garch)
h_test_gjr   = forecast_path_gjr(test["ret_pct"], gjr_fit, h0_test_gjr)

test_fcst = pd.DataFrame(index=test.index)
test_fcst["target_RV_next"] = test["target_RV_next"]

test_fcst["roll_36"] = df_model.loc[test.index, "roll_36"]
test_fcst["ewma_tuned"] = df_model.loc[test.index, "ewma_tuned_fcst"]

# Align forecasts as h_{t+1|t} stored at index t (available at bar close t)
r2_t = test["ret_pct"].pow(2)
test_fcst["garch11"] = garch_fit["omega"] + garch_fit["alpha"] * r2_t + garch_fit["beta"] * h_test_garch

ind_neg = (test["ret_pct"] < 0).astype(float)
test_fcst["gjr"] = (
    gjr_fit["omega"]
    + (gjr_fit["alpha"] + gjr_fit["gamma"] * ind_neg) * r2_t
    + gjr_fit["beta"] * h_test_gjr
)

if egarch_fit is not None:
    # EGARCH initialization for test: log(h_{t0}) from last train state
    h_last_eg = float(egarch_fit["h"].iloc[-1])
    logh_last = np.log(max(h_last_eg, 1e-12))
    z_last = r_last_train / np.sqrt(max(h_last_eg, 1e-12))
    Ez = np.sqrt(2.0 / np.pi)
    logh0_test = (
        egarch_fit["omega"]
        + egarch_fit["beta"] * logh_last
        + egarch_fit["alpha"] * (np.abs(z_last) - Ez)
        + egarch_fit["gamma"] * z_last
    )
    h0_test_eg = np.exp(logh0_test)

    h_test_eg = forecast_path_egarch(test["ret_pct"], egarch_fit, h0_test_eg)
    h_t = np.maximum(h_test_eg.values, 1e-12)
    z_t = test["ret_pct"].values / np.sqrt(h_t)
    logh_next = (
        egarch_fit["omega"]
        + egarch_fit["beta"] * np.log(h_t)
        + egarch_fit["alpha"] * (np.abs(z_t) - Ez)
        + egarch_fit["gamma"] * z_t
    )
    test_fcst["egarch"] = np.exp(logh_next)

test_fcst = test_fcst.dropna()

# Plot only top K models to reduce spaghetti
K_plot = min(4, len([c for c in test_fcst.columns if c != "target_RV_next"]))
# Rank by correlation with target for this preliminary view
prelim_corr = {c: test_fcst["target_RV_next"].corr(test_fcst[c]) 
               for c in test_fcst.columns if c != "target_RV_next"}
top_prelim = sorted(prelim_corr, key=lambda x: -prelim_corr[x])[:K_plot]

fig, ax = plt.subplots(figsize=(12, 4))
ax.plot(test_fcst.index, test_fcst["target_RV_next"], label="Target RV next", color="black", alpha=0.8)
for c in top_prelim:
    ax.plot(test_fcst.index, test_fcst[c], label=c, alpha=0.7)
ax.set_yscale("log")
ax.set_title(f"Out-of-sample one-step variance forecasts (top {K_plot} by corr, log scale)")
ax.set_ylabel("Variance (pct^2)")
ax.legend(ncol=3)
plt.show()

# %% [markdown]
#  ### 3.6 Diagnostics: standardized residuals
#  
#  After fitting, we check whether the model has removed most conditional heteroskedasticity by examining standardized residuals:
#  \[
#  z_t = \frac{r^{(\%)}_t}{\sqrt{h_t}}.
#  \]
#  
#  Diagnostics:
#  - Histogram of \(z_t\) to assess heavy tails and skewness.
#  - ACF of \(z_t^2\) to see whether volatility clustering remains after conditioning on the model.
# 

# %%
z_garch = train_raw["ret_pct"] / np.sqrt(garch_fit["h"])
z_gjr   = train_raw["ret_pct"] / np.sqrt(gjr_fit["h"])

fig, ax = plt.subplots(figsize=(12, 3))
ax.hist(z_garch, bins=60, density=True)
ax.set_title("Standardized residuals (GARCH) — heavy tails / skew show up here")
plt.show()

fig = plt.figure(figsize=(10, 4))
plot_acf((z_garch**2), lags=50, ax=plt.gca())
plt.title("ACF of squared standardized residuals (GARCH)")
plt.show()

# %% [markdown]
#  ## 4. Realized-vol tier (HAR‑RV)
#  
#  With high‑frequency data, forecasting **realized variance directly** often performs very well. Realized volatility is known to exhibit **long memory**: today’s volatility depends on volatility over multiple past horizons (short/medium/long).
# 
#  We implement a HAR‑style regression on **log realized variance**:
#  \[
#  \log RV_{t+1} = c + \beta_d \log RV^{(d)}_t + \beta_w \log RV^{(w)}_t + \beta_m \log RV^{(m)}_t + \varepsilon_{t+1}.
#  \]
#  
#  Practical choices:
#  - We use \(\log(RV+\varepsilon)\) to keep the target well‑behaved and reduce the influence of extreme spikes.
#  - Depending on how many days of data are available, we fit HAR on a **daily**, **hourly**, or **bar‑level** RV series with appropriately scaled lag windows.
#  - We optionally include a simple leverage proxy: a lagged indicator for a negative return.
# 

# %%
rv_daily = (ret_base**2).resample("1D").sum().dropna() * (RETURN_SCALE**2)
rv_hourly = (ret_base**2).resample("1H").sum().dropna() * (RETURN_SCALE**2)

if len(rv_daily) >= 30:
    rv_har = rv_daily.copy()
    har_name = "daily"
    lags = {"d": 1, "w": 5, "m": 22}
elif len(rv_hourly) >= 72:
    rv_har = rv_hourly.copy()
    har_name = "hourly"
    lags = {"d": 1, "w": 6, "m": 24}
else:
    rv_har = df["RV_pct2"].copy()
    har_name = f"bar({BAR_FREQ})"
    lags = {"d": 1, "w": 12, "m": 72}

print("HAR frequency:", har_name, "| points:", len(rv_har), "| lags:", lags)

eps = 1e-12
logrv = np.log(rv_har + eps)

X = pd.DataFrame(index=logrv.index)
X["logRV_d"] = logrv.shift(lags["d"])
X["logRV_w"] = logrv.rolling(lags["w"]).mean().shift(1)
X["logRV_m"] = logrv.rolling(lags["m"]).mean().shift(1)

# Leverage proxy: negative return on the same grid (if available)
if har_name.startswith("bar"):
    ret_har = df["ret_pct"].reindex(logrv.index)
elif har_name == "hourly":
    ret_har = (df["ret_pct"].resample("1H").sum()).reindex(logrv.index)
else:
    ret_har = (df["ret_pct"].resample("1D").sum()).reindex(logrv.index)

X["neg_ret"] = (ret_har < 0).astype(float).shift(1)

Y = logrv.shift(-1).rename("logRV_next")
har_df = pd.concat([Y, X], axis=1).dropna()

split_idx_har = int(len(har_df) * 0.7)
har_train = har_df.iloc[:split_idx_har]
har_test  = har_df.iloc[split_idx_har:]

X_train = sm.add_constant(har_train.drop(columns=["logRV_next"]))
Y_train = har_train["logRV_next"]
har_model = sm.OLS(Y_train, X_train).fit()
print(har_model.summary())

X_test = sm.add_constant(har_test.drop(columns=["logRV_next"]))
har_pred_log = har_model.predict(X_test)
har_pred_rv = np.exp(har_pred_log)

fig, ax = plt.subplots(figsize=(12, 4))
ax.plot(har_test.index, np.exp(har_test["logRV_next"]), label="Realized RV (next)", alpha=0.7)
ax.plot(har_test.index, har_pred_rv, label="HAR forecast", alpha=0.7)
ax.set_yscale("log")
ax.set_title(f"HAR-RV forecasts on {har_name} realized variance (log scale)")
ax.set_ylabel("RV (pct^2)")
ax.legend()
plt.show()

# %% [markdown]
#  ## 5. SV tier: Stochastic Volatility (latent log-vol) + regime switching
#  
#  GARCH models specify a deterministic recursion for variance given past returns. **Stochastic volatility (SV)** instead treats volatility as an unobserved state that evolves randomly. This often produces smoother volatility estimates and can capture changes in volatility dynamics in a natural way.
# 
#  ### 5.1 Approximate SV via a Gaussian state-space model (Kalman filter)
#  
#  A common SV specification is:
#  \[
#  r_t = \exp\left(\tfrac{1}{2}h_t\right)\epsilon_t,\qquad \epsilon_t \sim N(0,1),
#  \]
#  with latent log variance:
#  \[
#  h_t = \mu + \phi(h_{t-1}-\mu) + \sigma \eta_t,\qquad \eta_t \sim N(0,1).
#  \]
#  
#  For a fast, transparent baseline, we use a classic approximation based on log‑squared returns:
#  - define \(y_t = \log(r_t^2 + \varepsilon) - m\),
#  - under Gaussian SV, \(\log(\epsilon_t^2)\) has known mean \(m\) and variance \(V\),
#  - approximate \(y_t \approx h_t + e_t\) with \(e_t \sim N(0,V)\).
#  
#  This yields a linear Gaussian state‑space model that can be handled with the **Kalman filter**, and parameters \((\mu,\phi,\sigma)\) can be estimated by minimizing the (approximate) Gaussian negative log-likelihood.
#  
#  **Stability choices**
#  - Constrain persistence to \(0<\phi<1\) (volatility is typically positively persistent).
#  - Optimize over unconstrained transforms: \(\phi=\sigma(\theta)\) and \(\sigma=\exp(\theta)\).
#  - Use a small multi‑start set of initial values to reduce sensitivity to local minima.
# 

# %%
SV_M = -1.2704
SV_V = (np.pi**2) / 2

def sv_kalman_nll(y: np.ndarray, mu: float, phi: float, sigma: float, V: float = SV_V):
    T = len(y)
    a = mu
    P = sigma**2 / max(1e-12, (1.0 - phi**2))
    nll = 0.0
    for t in range(T):
        a_pred = mu + phi * (a - mu)
        P_pred = (phi**2) * P + sigma**2
        v = y[t] - a_pred
        S = P_pred + V
        nll += 0.5 * (np.log(2*np.pi) + np.log(S) + (v**2)/S)
        K = P_pred / S
        a = a_pred + K * v
        P = (1.0 - K) * P_pred
    return float(nll)

def sv_kalman_filter(y: np.ndarray, mu: float, phi: float, sigma: float, V: float = SV_V):
    T = len(y)
    a = mu
    P = sigma**2 / max(1e-12, (1.0 - phi**2))
    a_filt = np.empty(T)
    P_filt = np.empty(T)
    a_pred_arr = np.empty(T)
    P_pred_arr = np.empty(T)
    for t in range(T):
        a_pred = mu + phi*(a - mu)
        P_pred = (phi**2)*P + sigma**2
        v = y[t] - a_pred
        S = P_pred + V
        K = P_pred / S
        a = a_pred + K*v
        P = (1.0 - K)*P_pred
        a_filt[t] = a
        P_filt[t] = P
        a_pred_arr[t] = a_pred
        P_pred_arr[t] = P_pred
    return a_filt, P_filt, a_pred_arr, P_pred_arr

def fit_sv_approx_multistart(ret_pct: pd.Series, starts=None):
    r = ret_pct.values
    y = np.log(r**2 + 1e-12) - SV_M
    y = np.clip(y, -50, 50)

    if starts is None:
        mu0 = float(np.mean(y))
        starts = []
        for phi in [0.80, 0.90, 0.95, 0.98, 0.995]:
            for sigma in [0.05, 0.10, 0.20, 0.40]:
                starts.append((mu0, phi, sigma))

    best = None
    for mu0, phi0, sigma0 in starts:
        # optimize over (mu, logit_phi, log_sigma)
        x0 = np.array([mu0, np.log(phi0/(1-phi0)), np.log(sigma0)])
        bounds = [(-50, 50), (-10, 10), (-10, 3)]  # log_sigma lower bound keeps sigma >= ~4.5e-5

        def unpack(p):
            mu = p[0]
            phi = sigmoid(p[1]) * 0.999
            sigma = np.exp(p[2])
            return mu, phi, sigma

        def obj(p):
            mu, phi, sigma = unpack(p)
            return sv_kalman_nll(y, mu, phi, sigma, V=SV_V)

        res = minimize(obj, x0, method="L-BFGS-B", bounds=bounds)
        if not res.success:
            continue
        mu, phi, sigma = unpack(res.x)
        a_filt, P_filt, a_pred, P_pred = sv_kalman_filter(y, mu, phi, sigma, V=SV_V)
        cand = {
            "mu": mu, "phi": phi, "sigma": sigma,
            "success": True, "message": res.message, "nll": res.fun,
            "h_filt": pd.Series(a_filt, index=ret_pct.index),
            "P_filt": pd.Series(P_filt, index=ret_pct.index)
        }
        if best is None or cand["nll"] < best["nll"]:
            best = cand
    return best

sv_fit = fit_sv_approx_multistart(train["ret_pct"])
if sv_fit is None:
    print("SV approx: no stable fit found.")
else:
    print({k: sv_fit[k] for k in ["mu","phi","sigma","nll"]})
    sv_var_filt = np.exp(sv_fit["h_filt"])
    fig, ax = plt.subplots(figsize=(12, 4))
    ax.plot(train.index, sv_var_filt, label="SV approx exp(h_t) (filtered)")
    ax.set_yscale("log")
    ax.set_title("Approx SV filtered variance (log scale)")
    ax.set_ylabel("Variance (pct^2)")
    ax.legend()
    plt.show()

# %% [markdown]
#  ### 5.2 SV one-step-ahead variance forecasts on the test set
#  
#  After filtering the latent log variance, we produce one‑step‑ahead variance forecasts. Since \(h_{t+1}\) is Gaussian under the state equation, \(\exp(h_{t+1})\) is log‑normal and:
#  \[
#  \mathbb{E}\left[\exp(h_{t+1})\mid \mathcal{F}_t\right] = \exp\left(a_{t+1} + \tfrac{1}{2}P_{t+1}\right),
#  \]
#  where \(a_{t+1}\) and \(P_{t+1}\) are the predicted mean and variance of \(h_{t+1}\).  
#  This produces forecasts directly comparable to the next‑bar realized variance target.
# 

# %%
def sv_forecast_next_variance(ret_pct: pd.Series, params: dict):
    r = ret_pct.values
    y = np.log(r**2 + 1e-12) - SV_M
    y = np.clip(y, -50, 50)

    mu, phi, sigma = params["mu"], params["phi"], params["sigma"]
    V = SV_V

    T = len(y)
    a = mu
    P = sigma**2 / max(1e-12, (1.0 - phi**2))

    var_next = np.empty(T)
    for t in range(T):
        # predict h_t
        a_pred = mu + phi*(a - mu)
        P_pred = (phi**2)*P + sigma**2

        # update with y_t
        v = y[t] - a_pred
        S = P_pred + V
        K = P_pred / S
        a = a_pred + K*v
        P = (1.0 - K)*P_pred

        # next-step prediction moments for h_{t+1}
        a_next = mu + phi*(a - mu)
        P_next = (phi**2)*P + sigma**2

        # E[exp(h_{t+1})] for Gaussian h_{t+1}
        var_next[t] = np.exp(a_next + 0.5*P_next)

    return pd.Series(var_next, index=ret_pct.index)

if sv_fit is not None:
    sv_fcst_test = sv_forecast_next_variance(test["ret_pct"], sv_fit)
    test_fcst["sv_approx"] = sv_fcst_test
    test_fcst = test_fcst.dropna()

    # Plot only top K models to reduce spaghetti
    K_plot_sv = min(4, len([c for c in test_fcst.columns if c != "target_RV_next"]))
    sv_corr = {c: test_fcst["target_RV_next"].corr(test_fcst[c]) 
               for c in test_fcst.columns if c != "target_RV_next"}
    top_sv = sorted(sv_corr, key=lambda x: -sv_corr[x])[:K_plot_sv]
    
    fig, ax = plt.subplots(figsize=(12, 4))
    ax.plot(test_fcst.index, test_fcst["target_RV_next"], label="Target RV next", color="black", alpha=0.8)
    for c in top_sv:
        ax.plot(test_fcst.index, test_fcst[c], label=c, alpha=0.7)
    ax.set_yscale("log")
    ax.set_title(f"Test forecasts incl. approximate SV (top {K_plot_sv} by corr, log scale)")
    ax.set_ylabel("Variance (pct^2)")
    ax.legend(ncol=3)
    plt.show()

# %% [markdown]
#  ### 5.3 Regime switching (Markov switching variance)
#  
#  Volatility often alternates between “quiet” and “turbulent” periods. A simple way to capture this is a **Markov switching** model with regime‑dependent variance, where the regime follows a Markov chain.
#  
#  Here we fit a two‑regime specification, inspect smoothed regime probabilities, and compute the implied conditional variance as a probability‑weighted average of regime variances.
# 
# 
#  **Practical note:** regime-switching models typically need longer samples to estimate transition probabilities and regime variances reliably. With limited data, regimes may be weakly separated and probabilities can look noisy; treat this section as exploratory unless you have multiple days of observations.
# 

# %%
from statsmodels.tsa.regime_switching.markov_regression import MarkovRegression

mr = MarkovRegression(train["ret_pct"], k_regimes=2, trend="c", switching_variance=True)
mr_res = mr.fit(disp=False)
print(mr_res.summary())

probs = mr_res.smoothed_marginal_probabilities
fig, ax = plt.subplots(figsize=(12, 3))
ax.plot(probs.index, probs[0], label="Regime 0 prob")
ax.plot(probs.index, probs[1], label="Regime 1 prob")
ax.set_title("Smoothed regime probabilities (train)")
ax.set_ylabel("Probability")
ax.legend()
plt.show()

params = mr_res.params
var_keys = [k for k in params.index if "sigma2" in k or "variance" in k]
print("Variance keys:", var_keys)

if len(var_keys) >= 2:
    v0 = float(params[var_keys[0]])
    v1 = float(params[var_keys[1]])
    var_rs = probs[0]*v0 + probs[1]*v1
    fig, ax = plt.subplots(figsize=(12, 4))
    ax.plot(var_rs.index, var_rs, label="Regime-switching variance (smoothed)")
    ax.set_yscale("log")
    ax.set_title("Regime-switching conditional variance (log scale)")
    ax.set_ylabel("Variance (pct^2)")
    ax.legend()
    plt.show()

# %% [markdown]
#  ## 6. Compare properly: targets, losses, and out-of-sample evaluation
#  
#  We compare all models on the **test** period using the same aligned target:
#  \[
#  y_t = RV_{\%^2,\,t+1}.
#  \]
#  
#  Because realized variance is a noisy proxy, the choice of metric matters. We report:
#  - **MSE** on variance (sensitive to large spikes),
#  - **QLIKE** (commonly preferred for realized variance evaluation),
#  - **Correlation** (helpful intuition, but not a proper scoring rule).
# 
#  In addition to average losses, we plot **cumulative QLIKE** for the top few models. This view makes it easier to see whether a model is consistently better across time or only wins during a handful of extreme episodes.
# 

# %%
def mse(y_true: pd.Series, y_pred: pd.Series) -> float:
    a = pd.concat([y_true, y_pred], axis=1).dropna()
    return float(np.mean((a.iloc[:,0] - a.iloc[:,1])**2))

def qlike(y_true: pd.Series, y_pred: pd.Series, eps=1e-12) -> float:
    a = pd.concat([y_true, y_pred], axis=1).dropna()
    return qlike_loss(a.iloc[:,0], a.iloc[:,1], eps=eps)

models_to_eval = [c for c in test_fcst.columns if c != "target_RV_next"]
metrics = []
for m in models_to_eval:
    metrics.append({
        "model": m,
        "MSE": mse(test_fcst["target_RV_next"], test_fcst[m]),
        "QLIKE": qlike(test_fcst["target_RV_next"], test_fcst[m]),
        "corr": test_fcst["target_RV_next"].corr(test_fcst[m]),
    })

metrics_df = pd.DataFrame(metrics).sort_values("QLIKE")
metrics_df

# %% [markdown]
#  ### 6.1 Cumulative QLIKE (top models only)
#  
#  We compute per‑period QLIKE losses and cumulate them through the test sample. Lower cumulative loss indicates better performance under this scoring rule.
# 

# %%
loss_df = pd.DataFrame(index=test_fcst.index)
for m in models_to_eval:
    y = np.maximum(test_fcst["target_RV_next"].values, 1e-12)
    h = np.maximum(test_fcst[m].values, 1e-12)
    ratio = y / h
    loss_df[m] = ratio - np.log(ratio) - 1.0

# Choose top K by final mean QLIKE
K = min(6, len(models_to_eval))
top_models = metrics_df["model"].head(K).tolist()

cum = loss_df[top_models].cumsum()

fig, ax = plt.subplots(figsize=(12, 4))
for m in top_models:
    ax.plot(cum.index, cum[m], label=m)
ax.set_title("Cumulative QLIKE loss (top models; lower is better)")
ax.set_ylabel("Cumulative QLIKE")
ax.legend(ncol=3)
plt.show()

# %% [markdown]
#  ### 6.2 Forecast calibration scatter
#  
#  Finally, we visualize calibration for the best model by plotting forecast variance vs. realized variance (next bar) on a log–log scale. A well‑calibrated model should cluster around the 45‑degree line:
#  - points above the line indicate under‑prediction,
#  - points below the line indicate over‑prediction.
#  
#  A hexbin density plot reduces overplotting and highlights where most observations lie.
# 

# %%
best_model = metrics_df.iloc[0]["model"] if len(metrics_df) else None
if best_model is not None:
    # Use hexbin with log-log scale to avoid point pile-up near zero
    x = test_fcst[best_model].values
    y = test_fcst["target_RV_next"].values
    
    # Filter out zeros/negatives for log scale
    mask = (x > 0) & (y > 0)
    x_pos, y_pos = x[mask], y[mask]
    
    fig, ax = plt.subplots(figsize=(7, 6))
    hb = ax.hexbin(x_pos, y_pos, gridsize=30, cmap="YlOrRd", mincnt=1,
                   xscale="log", yscale="log", linewidths=0.2)
    
    # Add 45-degree reference line (perfect calibration)
    lims = [max(x_pos.min(), y_pos.min()), min(x_pos.max(), y_pos.max())]
    ax.plot(lims, lims, "k--", alpha=0.5, label="Perfect calibration")
    
    ax.set_title(f"Calibration hexbin: forecast vs realized (best={best_model})")
    ax.set_xlabel("Forecast variance (pct^2, log)")
    ax.set_ylabel("Realized variance next (pct^2, log)")
    plt.colorbar(hb, ax=ax, label="Count")
    ax.legend(loc="upper left")
    plt.show()




"""
Streamlit Stock Analysis + Chat Interface
Single-file Streamlit app that:
- fetches historical data with yfinance
- computes indicators (SMA, EMA, RSI, MACD, Bollinger, ATR, OBV, Stochastic)
- shows interactive Plotly charts
- provides a simple chat interface (rule-based + ML prediction) to ask about signals, predictions, and backtest

Save as `app.py` and run with:
    pip install -r requirements.txt
    streamlit run app.py

Requirements (requirements.txt):
streamlit
pandas
numpy
yfinance
scikit-learn
plotly
scipy

"""


import streamlit as st
import pandas as pd
import numpy as np
import yfinance as yf
from scipy import stats
from sklearn.ensemble import RandomForestClassifier
from sklearn.model_selection import TimeSeriesSplit
from sklearn.metrics import accuracy_score
import plotly.graph_objs as go
from datetime import datetime
import re

st.set_page_config(layout="wide", page_title="Stock Analysis Chat Bot")

# ---------------------- Utilities / Indicators ----------------------
@st.cache_data(show_spinner=False)
def fetch_history(ticker: str, period: str = "2y", interval: str = "1d") -> pd.DataFrame:
    data = yf.download(ticker, period=period, interval=interval, progress=False)
    if data is None or data.empty:
        return pd.DataFrame()
    
    # Fix MultiIndex columns if they exist
    if isinstance(data.columns, pd.MultiIndex):
        # Flatten MultiIndex columns by taking the first level
        data.columns = data.columns.get_level_values(0)
    
    # Ensure standard columns exist and copy
    required_cols = ['Open', 'High', 'Low', 'Close', 'Volume']
    available_cols = [col for col in required_cols if col in data.columns]
    
    if not available_cols:
        return pd.DataFrame()
        
    data = data[available_cols].copy()
    data.index = pd.to_datetime(data.index)
    
    # Ensure the index is not MultiIndex
    if isinstance(data.index, pd.MultiIndex):
        data.index = data.index.get_level_values(0)
    
    return data


def sma(series: pd.Series, window: int) -> pd.Series:
    return series.rolling(window).mean()


def ema(series: pd.Series, span: int) -> pd.Series:
    return series.ewm(span=span, adjust=False).mean()


def rsi(series: pd.Series, window: int = 14) -> pd.Series:
    delta = series.diff()
    up = delta.clip(lower=0)
    down = -1 * delta.clip(upper=0)
    # Use ewm smoothing
    ma_up = up.ewm(alpha=1/window, adjust=False).mean()
    ma_down = down.ewm(alpha=1/window, adjust=False).mean()
    rs = ma_up / ma_down
    rsi = 100 - (100 / (1 + rs))
    return rsi


def macd(series: pd.Series, span_short: int = 12, span_long: int = 26, span_signal: int = 9):
    ema_short = ema(series, span_short)
    ema_long = ema(series, span_long)
    macd_line = ema_short - ema_long
    signal = ema(macd_line, span_signal)
    hist = macd_line - signal
    return macd_line, signal, hist


def bollinger_bands(series: pd.Series, window: int = 20, n_std: float = 2.0):
    sma_ = sma(series, window)
    std_ = series.rolling(window).std()
    upper = sma_ + n_std * std_
    lower = sma_ - n_std * std_
    return upper, sma_, lower


def obv(df: pd.DataFrame):
    """
    On-Balance Volume (OBV).
    Implements robustly: ensures close and volume are Series and uses vectorized approach.
    """
    # Force Series (squeeze if needed)
    close = df['Close']
    volume = df['Volume']

    # Vectorized sign of price change
    # diff -> sign: +1 if up, -1 if down, 0 if unchanged
    sign = np.sign(close.diff()).fillna(0).astype(int)
    # Multiply sign by volume and cumsum
    obv_series = (sign * volume).cumsum()
    return obv_series.astype(float)  # ensure float dtype


def stochastic_oscillator(df: pd.DataFrame, k_window: int = 14, d_window: int = 3):
    low_min = df['Low'].rolling(k_window).min()
    high_max = df['High'].rolling(k_window).max()
    denom = (high_max - low_min)
    k = 100 * ((df['Close'] - low_min) / denom.replace({0: np.nan}))
    d = k.rolling(d_window).mean()
    return k, d


def linear_regression_slope(series: pd.Series, window: int = 30) -> pd.Series:
    """
    Compute rolling linear regression slope over a given window.
    Returns a Series aligned with the input index.
    Fixed version that handles array dimensions properly.
    """
    if len(series) < window:
        # If series is shorter than window, return all NaN
        return pd.Series([np.nan] * len(series), index=series.index)
    
    slopes = []
    
    # Create x values once (0, 1, 2, ..., window-1)
    x = np.arange(window)
    
    for i in range(len(series)):
        if i < window - 1:
            # Not enough data points yet
            slopes.append(np.nan)
        else:
            # Get the window of y values
            start_idx = i - window + 1
            end_idx = i + 1
            y = series.iloc[start_idx:end_idx].values
            
            # Check for NaN values in the window
            if np.isnan(y).any():
                slopes.append(np.nan)
            else:
                # Ensure y is 1-dimensional and same length as x
                y = y.flatten()  # Flatten to ensure 1D
                if len(y) != len(x):
                    slopes.append(np.nan)
                else:
                    try:
                        slope, _, _, _, _ = stats.linregress(x, y)
                        slopes.append(slope)
                    except Exception:
                        slopes.append(np.nan)
    
    # Ensure returned series has the same length as input
    return pd.Series(slopes, index=series.index)


def atr(df: pd.DataFrame, window: int = 14) -> pd.Series:
    """
    Compute Average True Range (ATR) with a rolling window.
    Fixed version that handles edge cases properly.
    """
    # Defensive check for sufficient data
    if df is None or df.empty or len(df) < 2:
        return pd.Series([np.nan] * len(df) if not df.empty else [], 
                        index=df.index if not df.empty else pd.Index([]))
    
    # Check if required columns exist
    required_cols = ['High', 'Low', 'Close']
    if not all(col in df.columns for col in required_cols):
        missing = [col for col in required_cols if col not in df.columns]
        raise ValueError(f"Missing required columns for ATR: {missing}")
    
    high, low, close = df['High'], df['Low'], df['Close']
    
    # Calculate True Range components
    tr1 = high - low
    tr2 = (high - close.shift(1)).abs()
    tr3 = (low - close.shift(1)).abs()
    
    # Use pandas concat and max to properly handle Series operations
    # This avoids both the scalar error and the 2D array issue
    true_range_df = pd.concat([tr1, tr2, tr3], axis=1)
    true_range = true_range_df.max(axis=1)
    
    # Calculate the rolling mean of the True Range
    atr_series = true_range.rolling(window=window, min_periods=1).mean()
    
    return atr_series

def add_all_indicators(df: pd.DataFrame) -> pd.DataFrame:
    if df is None or df.empty:
        return pd.DataFrame()
    df = df.copy()
    close = df['Close']
    df['SMA_20'] = sma(close, 20)
    df['SMA_50'] = sma(close, 50)
    df['EMA_12'] = ema(close, 12)
    df['EMA_26'] = ema(close, 26)
    df['RSI_14'] = rsi(close, 14)
    macd_line, macd_signal, macd_hist = macd(close)
    df['MACD'] = macd_line
    df['MACD_Signal'] = macd_signal
    df['MACD_Hist'] = macd_hist
    bb_upper, bb_mid, bb_lower = bollinger_bands(close, 20)
    df['BB_upper'] = bb_upper
    df['BB_mid'] = bb_mid
    df['BB_lower'] = bb_lower
    df['ATR_14'] = atr(df, 14)
    df['OBV'] = obv(df)
    k, d = stochastic_oscillator(df)
    df['Stoch_k'] = k
    df['Stoch_d'] = d
    df['LR_Slope_30'] = linear_regression_slope(close, 30)
    return df

# ---------------------- Signal logic & ML ----------------------

def moving_average_crossover_signals(df: pd.DataFrame, short_col: str = 'SMA_20', long_col: str = 'SMA_50'):
    s = df[short_col]
    l = df[long_col]
    prev_s = s.shift(1)
    prev_l = l.shift(1)
    signal = pd.Series(0, index=df.index)
    bullish = (prev_s < prev_l) & (s > l)
    bearish = (prev_s > prev_l) & (s < l)
    signal.loc[bullish.fillna(False)] = 1
    signal.loc[bearish.fillna(False)] = -1
    return signal


def macd_signal(df: pd.DataFrame):
    macd = df.get('MACD')
    sig = df.get('MACD_Signal')
    if macd is None or sig is None:
        return pd.Series(0, index=df.index)
    prev_macd = macd.shift(1)
    prev_sig = sig.shift(1)
    signal = pd.Series(0, index=df.index)
    signal.loc[((prev_macd < prev_sig) & (macd > sig)).fillna(False)] = 1
    signal.loc[((prev_macd > prev_sig) & (macd < sig)).fillna(False)] = -1
    return signal


def rsi_signal(df: pd.Series, low: int = 30, high: int = 70):
    rsi_s = df.get('RSI_14')
    if rsi_s is None:
        return pd.Series(0, index=df.index)
    prev = rsi_s.shift(1)
    signal = pd.Series(0, index=df.index)
    signal.loc[((prev < low) & (rsi_s >= low)).fillna(False)] = 1
    signal.loc[((prev > high) & (rsi_s <= high)).fillna(False)] = -1
    return signal


def trend_strength_by_slope(df: pd.DataFrame, slope_col='LR_Slope_30', threshold: float = 0.1):
    slope = df.get(slope_col)
    if slope is None:
        return pd.Series(0, index=df.index)
    out = slope.apply(lambda x: 1 if x > threshold else (-1 if x < -threshold else 0))
    return out


def aggregate_signals(df: pd.DataFrame):
    """
    Fixed version that uses completely different column names to avoid conflicts
    """
    signals = pd.DataFrame(index=df.index)
    
    # Use unique names that won't conflict with indicator columns
    signals['MA_Cross_Signal'] = moving_average_crossover_signals(df)
    signals['MACD_Cross_Signal'] = macd_signal(df)
    signals['RSI_Level_Signal'] = rsi_signal(df)
    signals['Trend_Slope_Signal'] = trend_strength_by_slope(df)
    
    # Update weights with new column names
    w = {
        'MA_Cross_Signal': 2.0, 
        'MACD_Cross_Signal': 1.5, 
        'RSI_Level_Signal': 1.0, 
        'Trend_Slope_Signal': 2.0
    }
    
    score = (signals['MA_Cross_Signal'] * w['MA_Cross_Signal'] +
             signals['MACD_Cross_Signal'] * w['MACD_Cross_Signal'] +
             signals['RSI_Level_Signal'] * w['RSI_Level_Signal'] +
             signals['Trend_Slope_Signal'] * w['Trend_Slope_Signal'])
    
    signals['Score'] = score
    signals['Decision'] = score.apply(lambda x: 'BUY' if x >= 2.0 else ('SELL' if x <= -2.0 else 'HOLD'))
    return signals

def prepare_ml_features(df: pd.DataFrame, horizon: int = 5):
    df = df.copy()
    df['future_close'] = df['Close'].shift(-horizon)
    df['return_future'] = (df['future_close'] - df['Close']) / df['Close']
    df['label'] = (df['return_future'] > 0).astype(int)
    feat_cols = ['Close', 'SMA_20', 'SMA_50', 'EMA_12', 'EMA_26',
                 'RSI_14', 'MACD', 'MACD_Signal', 'MACD_Hist',
                 'BB_upper', 'BB_mid', 'BB_lower', 'ATR_14', 'OBV',
                 'Stoch_k', 'Stoch_d', 'LR_Slope_30']
    # Only keep columns that actually exist
    feat_cols = [c for c in feat_cols if c in df.columns]
    X = df[feat_cols].dropna()
    y = df.loc[X.index, 'label']
    return X, y


def train_small_rf(X: pd.DataFrame, y: pd.Series):
    tscv = TimeSeriesSplit(n_splits=3)
    best_model = None
    scores = []
    for train_index, test_index in tscv.split(X):
        X_train, X_test = X.iloc[train_index], X.iloc[test_index]
        y_train, y_test = y.iloc[train_index], y.iloc[test_index]
        m = RandomForestClassifier(n_estimators=200, random_state=42)
        m.fit(X_train, y_train)
        preds = m.predict(X_test)
        scores.append(accuracy_score(y_test, preds))
    best_model = m
    return best_model, scores

# ---------------------- Backtest (simple) ----------------------

def simple_sma_backtest(df: pd.DataFrame, short_col='SMA_20', long_col='SMA_50', cash: float = 10000):
    df = df.copy().dropna(subset=[short_col, long_col])
    if df.empty:
        return pd.DataFrame()
    position = 0
    cash = float(cash)
    shares = 0
    history = []
    prev_short = df[short_col].shift(1)
    prev_long = df[long_col].shift(1)
    for date, row in df.iterrows():
        short = row[short_col]
        long = row[long_col]
        price = row['Close']
        # previous comparison guard (NaN won't satisfy)
        prev_ok = (date in prev_short.index) and (pd.notna(prev_short.loc[date])) and (pd.notna(prev_long.loc[date]))
        buy = prev_ok and (prev_short.loc[date] < prev_long.loc[date]) and (short > long)
        sell = prev_ok and (prev_short.loc[date] > prev_long.loc[date]) and (short < long)
        if buy and shares == 0 and price > 0:
            shares = int(cash // price)
            if shares > 0:
                cash -= shares * price
                history.append((date, 'BUY', price, shares))
        elif sell and shares > 0:
            cash += shares * price
            history.append((date, 'SELL', price, shares))
            shares = 0
        total = cash + shares * price
        history.append((date, 'VALUE', total, shares))
    if not history:
        return pd.DataFrame()
    perf = pd.DataFrame([{'date': d, 'type': t, 'price': p, 'shares': s} for (d, t, p, s) in history])
    perf.set_index('date', inplace=True)
    return perf

# ---------------------- Plotting Functions ----------------------

def plot_price(df: pd.DataFrame, ticker: str):
    fig = go.Figure()
    fig.add_trace(go.Candlestick(x=df.index, open=df['Open'], high=df['High'], low=df['Low'], close=df['Close'], name='Price'))
    if 'SMA_20' in df.columns:
        fig.add_trace(go.Scatter(x=df.index, y=df['SMA_20'], name='SMA 20', mode='lines'))
    if 'SMA_50' in df.columns:
        fig.add_trace(go.Scatter(x=df.index, y=df['SMA_50'], name='SMA 50', mode='lines'))
    if 'BB_upper' in df.columns and 'BB_lower' in df.columns:
        fig.add_trace(go.Scatter(x=df.index, y=df['BB_upper'], name='BB Upper', line={'dash':'dash'}))
        fig.add_trace(go.Scatter(x=df.index, y=df['BB_lower'], name='BB Lower', line={'dash':'dash'}))
    fig.update_layout(title=f"{ticker} Price & Indicators", xaxis_rangeslider_visible=False)
    return fig


def plot_indicator(df: pd.DataFrame, col: str, title: str = None):
    if col not in df.columns:
        return go.Figure()
    fig = go.Figure()
    fig.add_trace(go.Scatter(x=df.index, y=df[col], name=col))
    fig.update_layout(title=title or col)
    return fig

# ---------------------- Streamlit Layout ----------------------

st.sidebar.title("Configuration")
with st.sidebar.form(key='controls'):
    ticker = st.text_input('Ticker', value='AAPL')
    period = st.selectbox('Period', options=['6mo','1y','2y','5y'], index=2)
    interval = st.selectbox('Interval', options=['1d','1wk'], index=0)
    horizon = st.number_input('Prediction horizon (days)', value=5, min_value=1, max_value=30)
    run = st.form_submit_button('Fetch & Analyze')

if 'messages' not in st.session_state:
    st.session_state['messages'] = []

col1, col2 = st.columns([2,1])

with col1:
    st.title('Stock Analysis Chat Bot')
    if run:
        with st.spinner('Fetching data...'):
            df = fetch_history(ticker, period=period, interval=interval)
        if df.empty:
            st.error('No data found for that ticker/period. Try a different one.')
        else:
            df = add_all_indicators(df)
            signals = aggregate_signals(df)
            merged = df.join(signals, how='left')
            st.success(f'Data loaded: {len(df)} rows — last date {df.index[-1].date()}')
            # show price chart
            st.plotly_chart(plot_price(df, ticker), use_container_width=True)
            # RSI and MACD
            r1, r2 = st.columns(2)
            with r1:
                if 'RSI_14' in df.columns:
                    st.plotly_chart(plot_indicator(df, 'RSI_14', 'RSI (14)'), use_container_width=True)
            with r2:
                if 'MACD' in df.columns and 'MACD_Signal' in df.columns:
                    macd_df = df[['MACD','MACD_Signal']].dropna()
                    if not macd_df.empty:
                        fig = go.Figure()
                        fig.add_trace(go.Scatter(x=macd_df.index, y=macd_df['MACD'], name='MACD'))
                        fig.add_trace(go.Scatter(x=macd_df.index, y=macd_df['MACD_Signal'], name='Signal'))
                        st.plotly_chart(fig, use_container_width=True)

            # show last signal decision (guarded)
            last_signal = None
            if not signals['Decision'].dropna().empty:
                last_signal = signals['Decision'].dropna().iloc[-1]
                st.metric('Latest Decision', last_signal)
            else:
                st.metric('Latest Decision', 'N/A')

            # backtest summary
            perf = simple_sma_backtest(df)
            if not perf.empty:
                values = perf[perf['type']=='VALUE'] if 'type' in perf.columns else perf
                if not values.empty:
                    last_val = values.iloc[-1]['price'] if 'price' in values.columns else values.iloc[-1].get('total', np.nan)
                    st.write(f"Backtest snapshot (recent): {last_val:.2f}")

            # train small model
            X, y = prepare_ml_features(df, horizon=horizon)
            if len(X) > 120:
                with st.spinner('Training small ML model...'):
                    model, scores = train_small_rf(X, y)
                st.write('CV accuracies:', [round(s,3) for s in scores])
                # save model to session state
                st.session_state['rf_model'] = model
                st.session_state['rf_features'] = X.columns.tolist()
            else:
                st.info('Not enough rows to train ML model (need >120 rows of features).')

# ---------------------- Chat Interface ----------------------

# helper: safer ticker extraction from a message
_CMD_WORDS = {'signal','scan','predict','for','chart','plot','model','help','show','display','predicts','predicting','scan'}
def extract_ticker_from_text(msg: str, default: str = None):
    # look for typical ticker pattern: 1-5 letters, ignore common command words
    tokens = re.findall(r"\b[A-Za-z]{1,5}\b", msg)
    for t in tokens:
        t_up = t.upper()
        if t_up.lower() in _CMD_WORDS:
            continue
        # skip obvious English words by simple rule: skip if token length > 0 and token is in common words
        # (we already excluded _CMD_WORDS; user can still get false positives)
        if len(t_up) <= 5:
            return t_up
    return default

with col2:
    st.header('Chat')
    st.write('Ask the bot about signals, predictions, or commands like: "scan", "predict", "signal for AAPL"')
    user_input = st.text_input('You:', key='input')
    if st.button('Send') and user_input:
        st.session_state['messages'].append({'from':'user','text':user_input, 'time': datetime.utcnow().isoformat()})
        # simple intent detection
        msg = user_input.lower()
        response = "I couldn't understand that. Try: 'scan', 'signal', 'predict', or 'chart'."
        try:
            if 'signal' in msg or 'scan' in msg:
                t = extract_ticker_from_text(user_input, default=ticker)
                df2 = fetch_history(t, period=period, interval=interval)
                if df2.empty:
                    response = f'No data for {t}.'
                else:
                    df2 = add_all_indicators(df2)
                    signals2 = aggregate_signals(df2)
                    last_decision = signals2['Decision'].dropna()
                    last_score = signals2['Score'].dropna()
                    if not last_decision.empty and not last_score.empty:
                        last = last_decision.iloc[-1]
                        score = last_score.iloc[-1]
                        response = f'Latest decision for {t}: {last} (score {score:.2f})'
                    else:
                        response = f'No signals available for {t}.'
            elif 'predict' in msg or 'model' in msg:
                t = extract_ticker_from_text(user_input, default=ticker)
                df2 = fetch_history(t, period=period, interval=interval)
                if df2.empty:
                    response = f'No data for {t}.'
                else:
                    df2 = add_all_indicators(df2)
                    # Use stored features for prediction
                    if 'rf_model' in st.session_state and st.session_state['rf_model'] is not None:
                        model = st.session_state['rf_model']
                        feat_cols = st.session_state.get('rf_features', [])
                        
                        # Prepare the feature data from the new dataframe
                        Xp, _ = prepare_ml_features(df2, horizon=horizon)
                        
                        # Ensure the features for prediction match the training features
                        if not all(col in Xp.columns for col in feat_cols):
                            response = f'Cannot predict. Data for {t} is missing features required by the trained model. Try a different ticker or re-run "Fetch & Analyze" for {t}.'
                        else:
                            # Align the new data to the correct feature order
                            Xpred = Xp.tail(1)[feat_cols]
                            
                            if Xpred.isnull().any().any():
                                response = 'Latest feature row contains NaNs; cannot predict.'
                            else:
                                pred = model.predict(Xpred)[0]
                                prob = None
                                try:
                                    prob = model.predict_proba(Xpred)[0].max()
                                except Exception:
                                    prob = None
                                if prob is not None:
                                    response = f'Model predicts {"UP" if pred==1 else "DOWN"} with conf {prob:.2f}'
                                else:
                                    response = f'Model predicts {"UP" if pred==1 else "DOWN"}'
                    else:
                        response = 'No trained model available. Please press "Fetch & Analyze" first.'

            elif 'chart' in msg or 'plot' in msg:
                t = extract_ticker_from_text(user_input, default=ticker)
                df2 = fetch_history(t, period=period, interval=interval)
                if df2.empty:
                    response = f'No data for {t}.'
                else:
                    df2 = add_all_indicators(df2)
                    fig = plot_price(df2, t)
                    st.plotly_chart(fig, use_container_width=True)
                    response = f'Displayed chart for {t}.'
            elif 'help' in msg:
                response = "Try messages like: 'signal for AAPL', 'predict AAPL', 'chart TSLA', 'scan'."
            else:
                response = "Try commands: 'signal', 'predict', 'chart', or 'scan'."
        except Exception as e:
            response = f'Error processing request: {e}'
        st.session_state['messages'].append({'from':'bot','text':response, 'time': datetime.utcnow().isoformat()})

    # display conversation (most recent first)
    for m in reversed(st.session_state['messages']):
        if m['from']=='bot':
            st.markdown(f"**Bot:** {m['text']}")
        else:
            st.markdown(f"**You:** {m['text']}")

# ---------------------- Footer / tips ----------------------
st.sidebar.markdown('''
**Tips**
- Use ticker symbols (e.g. AAPL, TSLA, SPY).
- If charting or model prediction seems slow, reduce the period.
- This app uses yfinance; for intraday production systems use a paid quote provider.
''')
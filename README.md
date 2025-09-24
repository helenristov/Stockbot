
# Stock Analysis Chat Bot

A comprehensive Streamlit application that combines technical analysis, machine learning predictions, and an interactive chat interface for stock market analysis.

## Features

### Technical Analysis
- **Moving Averages**: Simple Moving Average (SMA) and Exponential Moving Average (EMA)
- **Momentum Indicators**: RSI (Relative Strength Index), Stochastic Oscillator
- **Trend Analysis**: MACD (Moving Average Convergence Divergence), Linear Regression Slope
- **Volatility**: Bollinger Bands, Average True Range (ATR)
- **Volume Analysis**: On-Balance Volume (OBV)

### Machine Learning
- Random Forest classifier for price direction prediction
- Time series cross-validation
- Feature engineering from technical indicators
- Confidence scoring for predictions

### Interactive Charts
- Candlestick price charts with overlaid indicators
- Separate panels for RSI and MACD visualization
- Responsive Plotly charts with zoom and pan functionality

### Chat Interface
- Natural language queries for stock analysis
- Commands: `signal`, `predict`, `chart`, `scan`
- Multi-ticker support
- Real-time signal analysis

### Backtesting
- Simple moving average crossover strategy
- Performance tracking with buy/sell signals
- Portfolio value monitoring

## Installation

### Requirements
```bash
pip install -r requirements.txt
```

**requirements.txt:**
```
streamlit
pandas
numpy
yfinance
scikit-learn
plotly
scipy
```

### Quick Start
```bash
git clone <repository-url>
cd stockbot
pip install -r requirements.txt
streamlit run streamlit_stock_bot.py
```

## Usage

### Main Interface

1. **Configure Analysis**
   - Enter ticker symbol (e.g., AAPL, TSLA, SPY)
   - Select time period (6mo, 1y, 2y, 5y)
   - Choose interval (1d, 1wk)
   - Set prediction horizon (1-30 days)

2. **Fetch & Analyze**
   - Click "Fetch & Analyze" to load data and compute indicators
   - View interactive price charts with technical indicators
   - Check latest trading signals and decisions

3. **Machine Learning Model**
   - Model trains automatically with sufficient data (>120 rows)
   - View cross-validation accuracy scores
   - Model used for chat predictions

### Chat Commands

#### Signal Analysis
```
signal for AAPL
scan TSLA
```
Returns latest buy/sell/hold decision with confidence score.

#### Price Predictions
```
predict AAPL
model prediction for TSLA
```
Uses trained ML model to predict price direction (UP/DOWN) with confidence.

#### Chart Display
```
chart AAPL
plot TSLA
```
Displays interactive price chart with technical indicators.

#### Help
```
help
```
Shows available commands and usage examples.

## Technical Indicators

### Moving Averages
- **SMA 20/50**: Simple moving averages for trend identification
- **EMA 12/26**: Exponential moving averages for MACD calculation

### Momentum Indicators
- **RSI (14)**: Identifies overbought (>70) and oversold (<30) conditions
- **Stochastic %K/%D**: Momentum oscillator comparing closing price to price range

### Trend Indicators
- **MACD**: Shows relationship between two moving averages
- **Linear Regression Slope**: Quantifies trend strength and direction

### Volatility Indicators
- **Bollinger Bands**: Price channels based on standard deviation
- **ATR (14)**: Measures market volatility

### Volume Indicators
- **OBV**: Cumulative volume indicator showing money flow

## Signal Generation

The application generates trading signals using weighted combinations of:

- **Moving Average Crossovers** (Weight: 2.0)
- **MACD Crossovers** (Weight: 1.5)
- **RSI Levels** (Weight: 1.0)
- **Trend Slope** (Weight: 2.0)

**Decision Thresholds:**
- BUY: Score ≥ 2.0
- SELL: Score ≤ -2.0
- HOLD: -2.0 < Score < 2.0

## Machine Learning Model

### Features Used
- Price data (Close, Open, High, Low)
- All technical indicators
- Volume data

### Model Details
- **Algorithm**: Random Forest Classifier (200 trees)
- **Validation**: Time Series Cross-Validation (3 folds)
- **Target**: Binary classification (price up/down)
- **Horizon**: User-configurable (1-30 days)

### Performance
- Model accuracy displayed after training
- Confidence scores provided with predictions
- Automatic retraining with new data

## File Structure

```
stockbot/
├── streamlit_stock_bot.py    # Main application file
├── requirements.txt          # Python dependencies
└── README.md                # This file
```

## Troubleshooting

### Common Issues

**No data found for ticker:**
- Verify ticker symbol is correct
- Try different time period
- Check if market is open/symbol exists

**Insufficient data for ML model:**
- Use longer time period (2y or 5y recommended)
- Some indicators require minimum data points

**Chart not displaying:**
- Refresh the page
- Check browser console for errors
- Ensure all dependencies are installed

### Performance Tips

- Use shorter time periods for faster loading
- Weekly intervals process faster than daily
- Reduce prediction horizon for quicker training

## Data Source

This application uses [yfinance](https://github.com/ranaroussi/yfinance) for historical stock data. Note that:

- Data is delayed (not real-time)
- For production systems, consider paid data providers
- Rate limits may apply for frequent requests

## Limitations

- **Educational Purpose**: Not intended for actual trading decisions
- **Data Delays**: Historical data only, not real-time
- **No Guarantees**: Past performance doesn't predict future results
- **Simplified Backtesting**: Basic strategy implementation only

## License

This project is for educational and demonstration purposes. Users are responsible for their own trading decisions and should consult with financial professionals before making investment choices.

## Contributing

To contribute to this project:

1. Fork the repository
2. Create a feature branch
3. Make your changes
4. Test thoroughly
5. Submit a pull request

## Support

For issues or questions:
- Check the troubleshooting section
- Review error messages in the Streamlit interface
- Ensure all dependencies are properly installed

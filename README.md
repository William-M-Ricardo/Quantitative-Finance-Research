# Trading Strategy Demo 
## Author: WilliamRichardXiao
## Email: xiaozhentao@mail.nwpu.edu.cn
This repository contains a collection of quantitative trading strategies implemented. The project includes both traditional CTA strategies and modern machine learning approaches for financial market prediction and trading.

## Project Structure

```
Trading Strategy Demo/
├── CTA Optimization Strategy/
│   ├── atr_rsi_demo.py
│   ├── aegis_strategy.py
│   ├── surge_strategy.py
│   ├── IF99_CFFEX.csv
│   ├── rb99_SHFE.csv
│   ├── atr_rsi_backtesting.ipynb
│   ├── aegis_backtesting.ipynb
│   └── surge_backtesting.ipynb
└── Machine Learning Trading Strategy/
    ├── dataset.pkl
    ├── research_workflow_xgb.ipynb
    ├── research_workflow_lstm.ipynb
    ├── xgb_model.py
    └── lstm_model.py
```

## Strategies Overview

### CTA Optimization Strategies

1. **ATR-RSI Strategy (`atr_rsi_demo.py`)**
   - Combines Average True Range (ATR) and Relative Strength Index (RSI) indicators
   - Uses ATR for trend strength identification and RSI for entry signals
   - Implements trailing stop loss for risk management

2. **Aegis Strategy (`aegis_strategy.py`)**
   - Integrates Bollinger Bands, Aroon oscillator, and ATR indicators
   - Uses Bollinger Bands for volatility breakout detection
   - Aroon oscillator for trend direction identification
   - ATR for position sizing and risk management

3. **Surge Strategy (`surge_strategy.py`)**
   - Based on intraday range breakout and RSI signals
   - Uses opening range breakout methodology with k1/k2 coefficients
   - Incorporates RSI for signal filtering
   - Features adaptive trailing stops for long and short positions

### Machine Learning Strategies

1. **XGBoost Model (`xgb_model.py`)**
   - Implements gradient boosting algorithm for factor prediction
   - Uses technical indicators as features (Alpha158 factors)
   - Provides feature importance analysis through weight and gain metrics
   - Includes hyperparameter tuning capabilities

2. **LSTM Model (`lstm_model.py`)**
   - Deep learning model using Long Short-Term Memory networks
   - Designed for sequential financial data processing
   - Features multiple LSTM layers with dropout regularization
   - Includes feature importance analysis through perturbation testing

## Installation Guide

1. **Prerequisites**
   - Python 3.7 or higher
   - VNPY platform installed
   - Required packages:
     ```bash
     pip install numpy pandas polars scikit-learn xgboost torch plotly
     ```

2. **Setup Instructions**
   ```bash
   # Clone the repository
   git clone <repository-url>
   
   # Navigate to project directory
   cd trading-strategy-demo
   
   # Install dependencies
   pip install -r requirements.txt(Currently None)
   ```

## Usage Instructions

### CTA Strategy Backtesting

1. Open the desired backtesting notebook (e.g., `atr_rsi_backtesting.ipynb`)
2. Configure the strategy parameters in the notebook
3. Run all cells to execute backtesting
4. View performance metrics and equity curves

### Machine Learning Workflow

1. **Data Preparation**
   - Run `research_workflow_xgb.ipynb` or `research_workflow_lstm.ipynb`
   - Load and preprocess financial data
   - Calculate technical indicators and features

2. **Model Training**
   - Execute model training cells
   - Monitor training/validation metrics
   - Analyze feature importance

3. **Prediction and Evaluation**
   - Generate predictions on test data
   - Evaluate model performance
   - Visualize results

## Strategy Parameters

### CTA Strategies
- **ATR-RSI Strategy**: atr_window, rsi_window, rsi_entry, trailing_percent
- **Aegis Strategy**: boll_window, boll_dev, aroon_window, atr_window
- **Surge Strategy**: k1, k2, rsi_window, trailing_long, trailing_short

### ML Strategies
- **XGBoost**: learning_rate, max_depth, reg_alpha, reg_lambda
- **LSTM**: hidden_size, num_layers, dropout, learning_rate

## Performance Metrics

Strategies are evaluated using standard quantitative metrics:
- Total Return
- Annualized Return
- Maximum Drawdown
- Sharpe Ratio
- Win Rate
- Profit Factor

## Contributing

1. Fork the repository
2. Create a feature branch
3. Commit your changes
4. Push to the branch
5. Create a pull request

## License

This project is licensed under the MIT License - see the LICENSE file for details.

## Acknowledgments

- Built using the VNPY quantitative trading platform
- Inspired by classical technical analysis and modern machine learning approaches
- Data sourced from financial market providers
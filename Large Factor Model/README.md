# Stock Momentum and Portfolio Optimization Project

This project analyzes stock momentum strategies and implements portfolio optimization techniques using machine learning models. It's designed to process financial data, calculate momentum signals, and build managed portfolios with risk management.

## Features

- Financial data processing with pandas
- Momentum signal calculation and analysis
- Portfolio construction and optimization
- Risk management techniques implementation
- Neural network models for portfolio prediction
- Multi-head attention mechanisms for stock analysis

## Requirements

- Python 3.12
- pandas
- numpy
- seaborn
- matplotlib
- PyTorch
- statsmodels
- tqdm

Install dependencies with:
```bash
pip install -r requirements.txt
```

## Project Structure

- `main.py`: Main implementation file with all analysis and modeling code
- `deploy.ipynb`: Jupyter notebook with saved execution results
- `requirements.txt`: Python package dependencies

## Key Components

1. **Data Processing**: Loads stock data from pickle file and processes it by size groups
2. **Momentum Analysis**: Calculates momentum signals and managed returns
3. **Risk Management**: Implements volatility management techniques
4. **Portfolio Optimization**: Uses ridge regression for efficient portfolio construction
5. **Neural Networks**: Implements MLP and Multi-Head Attention models for portfolio prediction
6. **Performance Evaluation**: Calculates Sharpe ratios and other performance metrics

## Usage

Run the main analysis:
```bash
python main.py
```

View the notebook results:
```bash
jupyter notebook deploy.ipynb
```

## Methodology

The project follows these main steps:
1. Load and preprocess stock data
2. Calculate momentum signals
3. Build managed portfolios
4. Apply risk management techniques
5. Optimize portfolios using ridge regression
6. Implement and train neural network models
7. Evaluate performance using Sharpe ratios

## Results

The project generates various visualizations including:
- Cumulative returns plots
- Sharpe ratio calculations
- Attention mechanism visualizations
- Portfolio performance comparisons


# Random Features for Stock Market Factors

## Project Overview

This is a research Jupyter Notebook focused on stock market factor construction and portfolio optimization. Based on the paper **"APT or AIPT: The Surprising Dominance of Large Factor Models"** by Antoine Didisheim, Barry Ke, Bryan Kelly, and Semyon Malamud, this research explores the use of random features and nonlinear transformations for constructing stock market factors.

## Research Background

### Data Source
- **Dataset**: From "Is There a Replication Crisis in Finance?" (https://jkpfactors.com/)
- **Time Period**: January 31, 1963 to December 31, 2019
- **Number of Stocks**: Approximately 3,000-4,000 US stocks per month
- **Feature Count**: 153 stock characteristics, reduced to 132 after preprocessing
- **Frequency**: Monthly data

### Data Preprocessing
1. **Feature Selection**: Select 132 characteristics with the smallest percentage of missing values
2. **Standardization**: Cross-sectionally rank each characteristic and normalize to the [-0.5, 0.5] range
3. **Missing Value Handling**: Fill remaining missing values with zeros

## Core Methodologies

### 1. Momentum Factor Construction
The research implements various momentum strategies:
- **Simple Momentum**: Based on returns over different time windows (12-month, 3-month, 6-month, 9-month)
- **Risk-Adjusted Momentum**: Using "Risk Managed Momentum" method to adjust weights based on historical volatility
- **Benchmark Comparison**: Performance comparison with market portfolio

### 2. Random Features Method
Implements random feature transformations to construct nonlinear factors:

```
F_t(γ) = concatenate(cos(X_t W_g), sin(X_t W_g))
F_{t+1} = R_{t+1}' S_t(γ)
```

Where:
- `X_t`: N_t × d feature matrix
- `W_g`: d × (P/2) random weight matrix
- `γ`: Scale parameter controlling nonlinearity

### 3. Portfolio Optimization
- **Ridge Regression**: Efficient matrix decomposition methods for regularized regression
- **Efficient Portfolio**: Optimal portfolio construction based on Markowitz theory
- **Rolling Window**: Out-of-sample testing to evaluate strategy performance

### 4. Deep Learning Methods
Implements various neural network architectures:

#### Flexible MLP Network
- Configurable number of layers and neurons
- LeCun initialization
- Custom loss function (MSRR - Maximal Sharpe Ratio Regression)

#### Multi-Head Attention Mechanism
Extended multi-head attention network implementation:
- **Query, Key, Value Projections**: X × W^Q, X × W^K, X × W^V
- **Attention Weights**: softmax(QK^T/√d)
- **Multi-Head Output Concatenation**: Concat(head_1, ..., head_h)
- **Final Output**: Linear projection to scalar predictions

## Technical Implementation

### Key Dependencies
- **Python**: Core programming language
- **NumPy**: Numerical computing
- **Pandas**: Data processing and analysis
- **PyTorch**: Deep learning framework
- **Matplotlib/Seaborn**: Data visualization
- **Statsmodels**: Statistical analysis and hypothesis testing

### Key Functions
1. `build_managed_returns()`: Construct managed returns
2. `sharpe_ratio()`: Calculate Sharpe ratio
3. `ridge_regr()`: Efficient ridge regression implementation
4. `efficient_portfolio_oos()`: Out-of-sample efficient portfolio
5. `produce_random_feature_managed_returns()`: Generate random feature factors
6. `volatility_managed_returns()`: Volatility management

### Evaluation Metrics
- **Sharpe Ratio**: Risk-adjusted returns
- **t-statistics**: Statistical significance with Newey-West standard errors
- **Cumulative Return Curves**: Visual representation of strategy performance
- **Correlation Analysis**: Correlation with benchmark strategies

## Research Findings

### 1. Momentum Strategy Performance
- Risk-adjusted momentum strategies significantly outperform simple market portfolio
- Momentum risk is highly predictable
- Volatility management effectively improves strategy stability

### 2. Random Features Effectiveness
- Random features can capture nonlinear relationships
- Large-scale factor models show excellent performance
- Nonlinear features provide incremental information to original linear models

### 3. Deep Learning Results
- MLP networks can effectively learn complex nonlinear relationships
- Attention mechanisms provide modeling capabilities for inter-stock relationships
- Parameter initialization is crucial for model convergence

### 4. Model Comparisons
- Large-scale factor models show excellent out-of-sample performance
- Risk management strategies significantly improve robustness
- Attention mechanisms demonstrate potential for cross-stock information utilization

## Code Structure

### Main Modules
1. **Data Loading and Preprocessing**
   - Data reading and cleaning
   - Feature standardization
   - Dataset splitting

2. **Factor Construction Module**
   - Momentum factor calculation
   - Random feature generation
   - Factor combination strategies

3. **Portfolio Optimization**
   - Ridge regression models
   - Efficient portfolio calculation
   - In-sample and out-of-sample evaluation

4. **Deep Learning Module**
   - MLP network implementation
   - Multi-head attention mechanism
   - Custom loss functions

### Experimental Design
- **Time Series Split**: 1963-2000 as training set, 2000-2019 as test set
- **Rolling Window**: Use historical data to predict future returns
- **Multiple Benchmarks**: Comparison with market portfolio, simple momentum strategies

## Usage Instructions

### Environment Setup
```bash
pip install numpy pandas torch matplotlib seaborn statsmodels
```

### Data Preparation
1. Download dataset to designated folder
2. Modify path configuration
3. Run data preprocessing code

### Workflow
1. Data loading and exploratory analysis
2. Basic momentum factor construction
3. Risk-adjusted strategy implementation
4. Random features method application
5. Deep learning model training
6. Result analysis and comparison

## Conclusions and Future Work

### Key Contributions
1. Complete factor construction and portfolio optimization pipeline implementation
2. Validation of random features effectiveness in financial factor construction
3. Exploration of deep learning methods in portfolio applications
4. Practical risk management framework provision

### Future Research Directions
1. Extension to more asset classes and markets
2. Integration of fundamental and alternative data
3. Real-time transaction cost considerations
4. More complex deep learning architectures

### Practical Value
- Technical framework provision for quantitative investment
- Demonstration of machine learning applications in financial forecasting
- Practical risk management tools
- Complete experimental platform for academic research

## Important Notes

1. **Data Sensitivity**: Financial data requires strict data governance and compliance handling
2. **Model Validation**: Requires rigorous out-of-sample testing and risk control
3. **Computational Resources**: Large-scale factor models require sufficient computational resources
4. **Risk Control**: Practical applications need to consider transaction costs, liquidity factors

## Acknowledgments

This research references important literature:
- "APT or AIPT: The Surprising Dominance of Large Factor Models"
- "Is There a Replication Crisis in Finance?"
- "Momentum Has Its Moments"
- "Artificial Intelligence Pricing Models"

Thanks to the open-source community for providing tools and data support.
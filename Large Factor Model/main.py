import numpy as np
import pandas as pd
import seaborn as sns
import matplotlib.pyplot as plt
import math
import statsmodels.api as sm
import torch                     # for all things PyTorch
import torch.nn as nn            # for torch.nn.Module, the parent object for PyTorch models
import torch.nn.functional as F  # for the activation function
import random

stock_data = pd.read_pickle('usa_131_per_size_ranks_False.pkl')
size_group = 'mega'

if size_group is not None:
  stock_data = stock_data.loc[stock_data.size_grp==size_group]

stock_data.set_index(['id', 'date'], inplace=True)
size_groups = stock_data.pop('size_grp')

# the simplest are momentum signals
momentum_columns = [x for x in stock_data.columns if 'ret' in x]

def build_managed_returns(returns, signals):
  # I am using numpy broadcasting here
  managed_returns = (signals * returns.values.reshape(-1, 1)).groupby(signals.index.get_level_values('date')).sum()
  return managed_returns

momentum_managed_returns = build_managed_returns(returns=stock_data['r_1'], signals=stock_data[momentum_columns])

# get stock data on that particular date
dat = stock_data.loc[stock_data.index.get_level_values('date')=='2022-11-30']
# let us see the magic of numpy broadcasting
tmp = dat['r_1'].values.reshape(-1, 1) * dat[momentum_columns]

factor_returns_on_the_date = tmp.sum()
factor2 = momentum_managed_returns.loc['2022-11-30']
tmp1 = pd.concat([factor_returns_on_the_date, factor2], axis=1)

def sharpe_ratio(returns):
  """
  The data is at monthly frequency, hence we multiply by sqrt(12)
  """
  return np.round(np.sqrt(12) * returns.mean() / returns.std(), 2)

momentum_managed_returns.cumsum().plot()
sr = sharpe_ratio(momentum_managed_returns)
plt.title(f'{sr}')
plt.show()

momentum_managed_returns.loc['2001-01-01':].cumsum().plot()
sr = sharpe_ratio(momentum_managed_returns.loc['2001-01-01':])
plt.title(f'{sr}')
plt.show()

simple_benchmark = (momentum_managed_returns * np.sign(momentum_managed_returns.shift(1).expanding().sum())).mean(1)
simple_benchmark.loc['1995-01-01':].cumsum().plot()
sr = sharpe_ratio(simple_benchmark.loc['1995-01-01':])
plt.title(f'{sr}')
plt.show()

def volatility_managed_returns(rets, window):
  return rets / rets.rolling(window).std().shift(1)

# let us do risk-managed momentum following "Momentum has its moments"
managed = volatility_managed_returns(momentum_managed_returns, 12)
simple_managed_benchmark = (managed * np.sign(managed.shift(1).expanding().sum())).mean(1)
managed['mean'] = simple_managed_benchmark
managed.cumsum().plot()
sr = sharpe_ratio(managed)
plt.title(f'{sr}')
plt.show()

market = stock_data.r_1.groupby('date').mean()
tmp = pd.concat([market, simple_benchmark, simple_managed_benchmark], axis=1)
print(tmp.corr())

def ridge_regr(signals: np.ndarray,
                  labels: np.ndarray,
                  future_signals: np.ndarray,
                  shrinkage_list: np.ndarray):
    """
    Regression is
    beta = (zI + S'S/t)^{-1}S'y/t = S' (zI+SS'/t)^{-1}y/t
    Inverting matrices is costly, so we use eigenvalue decomposition:
    (zI+A)^{-1} = U (zI+D)^{-1} U' where UDU' = A is eigenvalue decomposition,
    and we use the fact that D @ B = (diag(D) * B) for diagonal D, which saves a lot of compute cost
    :param signals: S
    :param labels: y
    :param future_signals: out of sample y
    :param shrinkage_list: list of ridge parameters
    :return:
    """
    t_ = signals.shape[0]
    p_ = signals.shape[1]
    if p_ < t_:
        # this is standard regression
        eigenvalues, eigenvectors = np.linalg.eigh(signals.T @ signals / t_)
        means = signals.T @ labels.reshape(-1, 1) / t_
        multiplied = eigenvectors.T @ means
        intermed = np.concatenate([(1 / (eigenvalues.reshape(-1, 1) + z)) * multiplied for z in shrinkage_list],
                                  axis=1)
        betas = eigenvectors @ intermed
    else:
        # this is the weird over-parametrized regime
        eigenvalues, eigenvectors = np.linalg.eigh(signals @ signals.T / t_)
        means = labels.reshape(-1, 1) / t_
        multiplied = eigenvectors.T @ means # this is \mu

        # now we build [(z_1+\delta)^{-1}, \cdots, (z_K+\delta)^{-1}] * \mu
        intermed = np.concatenate([(1 / (eigenvalues.reshape(-1, 1) + z)) * multiplied for z in shrinkage_list],
                                  axis=1)

        tmp = eigenvectors.T @ signals # U.T @ S
        betas = tmp.T @ intermed # (S.T @ U) @ [(z_1+\delta)^{-1}, \cdots, (z_K+\delta)^{-1}] * \mu
    predictions = future_signals @ betas
    return betas, predictions

def efficient_portfolio_oos(raw_factor_returns: pd.DataFrame):
  """

  """

  split = int(raw_factor_returns.shape[0] / 2)
  in_sample = raw_factor_returns.iloc[:split, :].values
  shrinkage_list = [0.00000001, 0.000001, 0.00001, 0.0001, 0.001, 0.01, 0.1, 1, 10, 100, 1000, 10000]
  _, optimal = ridge_regr(signals=in_sample,
                       labels=np.ones([in_sample.shape[0], 1]),
                       future_signals=raw_factor_returns.iloc[split:],
                       shrinkage_list=shrinkage_list)
  optimal.columns = shrinkage_list
  (optimal / optimal.std()).cumsum().plot()
  plt.title(f'{sharpe_ratio(optimal)}')
  return optimal, split

optimal, split = efficient_portfolio_oos(raw_factor_returns=managed.fillna(0))#momentum_managed_returns)

def regression_with_tstats(predicted_variable, explanatory_variables):
    x_ = explanatory_variables
    x_ = sm.add_constant(x_)
    y_ = predicted_variable
    # Newey-West standard errors with maxlags
    z_ = x_.copy().astype(float)
    result = sm.OLS(y_.values, z_.values).fit(cov_type='HAC', cov_kwds={'maxlags': 10})
    try:
        tstat = np.round(result.summary2().tables[1]['z'], 1)  # alpha t-stat (because for 'const')
        tstat.index = list(z_.columns)
    except:
        print(f'something is wrong for t-stats')
    return tstat

tstats = pd.concat([regression_with_tstats(predicted_variable=optimal[col],
                                           explanatory_variables=simple_managed_benchmark.fillna(0).iloc[split:]) for col in optimal], axis=1)
tstats.columns = optimal.columns

def produce_random_feature_managed_returns(P, stock_data, signals, num_seeds=2):
  """
  Suppose I wanted to build P random features. Using the weights \Theta \in \R^{d\times P},
  I could just do signals @ \Theta. If signals are (NT)\times d dimensional, then
  signals @ \Theta are (NT) \times P dimensional.
  Inststead, we can generate random features in small chunks, compute factors and proceed further.
  """
  all_random_feature_managed_returns = pd.DataFrame()
  d = signals.shape[1]
  for seed in range(num_seeds):
    # every seed gives me a new chunk of factors
    np.random.seed(seed)
    omega = scale * np.sqrt(2) * np.random.randn(P, d) / np.sqrt(d)
    ins_sin = np.sqrt(2) * np.sin(signals @ omega.T) # signals @ \Theta are (NT) \times P dimensional.
    ins_cos = np.sqrt(2) * np.cos(signals @ omega.T) # signals @ \Theta are (NT) \times P dimensional.
    random_features = pd.concat([ins_sin, ins_cos], axis=1)

    # Now, I collapse the N dimension.
    random_feature_managed_returns = build_managed_returns(returns=stock_data['r_1'], signals=random_features)
    # random_feature_managed_returns are now T \times P
    all_random_feature_managed_returns = pd.concat([all_random_feature_managed_returns, random_feature_managed_returns], axis=1)
  return all_random_feature_managed_returns

signals = stock_data[momentum_columns]
P =  100
d = signals.shape[1] # d=6 momentum signals
scale = 1.
random_feature_managed_returns = produce_random_feature_managed_returns(P, stock_data, signals, num_seeds=10)

optimal_random_features, split = efficient_portfolio_oos(raw_factor_returns=random_feature_managed_returns)

# does risk management help
vol_managed_rf = volatility_managed_returns(optimal_random_features, 12)
vol_managed_rf.cumsum().plot()
sr = sharpe_ratio(vol_managed_rf)
plt.title(f'{sr}')
plt.show()

# what if we manage each of them individually:
individually_managed = volatility_managed_returns(random_feature_managed_returns, 12).dropna()
optimal_random_features_individually_managed, split = efficient_portfolio_oos(raw_factor_returns=individually_managed)

tstats = pd.concat([regression_with_tstats(predicted_variable=optimal_random_features[col].fillna(0),
                                           explanatory_variables=pd.concat([simple_managed_benchmark.fillna(0).iloc[split:], optimal[10]], axis=1).fillna(0).reindex(optimal_random_features[col].index).fillna(0)) for col in optimal], axis=1)
tstats.columns = optimal.columns

tstats = pd.concat([regression_with_tstats(predicted_variable=vol_managed_rf[col].fillna(0),
                                           explanatory_variables=pd.concat([simple_managed_benchmark.fillna(0).iloc[split:], optimal[10]], axis=1).fillna(0).reindex(vol_managed_rf[col].index).fillna(0)) for col in vol_managed_rf], axis=1)
tstats.columns = optimal.columns

tstats = pd.concat([regression_with_tstats(predicted_variable=optimal_random_features_individually_managed[col].fillna(0),
                                           explanatory_variables=pd.concat([simple_benchmark.fillna(0).iloc[split:], optimal[10]], axis=1).fillna(0).reindex(optimal_random_features_individually_managed.index).fillna(0)) for col in optimal_random_features_individually_managed], axis=1)
tstats.columns = optimal.columns

signals = stock_data[momentum_columns]
labels = stock_data['r_1']
date_split = '2000-01-01'

train_signals = signals.loc[(signals.index.get_level_values('date') <= '2000-01-01') & (signals.index.get_level_values('date') >= '1970-01-01')]
train_returns = stock_data['r_1'].loc[(signals.index.get_level_values('date') <= '2000-01-01') & (signals.index.get_level_values('date') >= '1970-01-01')]

test_signals = signals.loc[signals.index.get_level_values('date') > '2000-01-01']
test_returns = stock_data['r_1'].loc[signals.index.get_level_values('date') > '2000-01-01']

def train_loader(signals, returns):
  """
  This is a special DataLoader designed to work with portfolio optimization.
  It creates mini-batches using every month of data
  """
  dates = signals.index.get_level_values('date')
  unique_dates = dates.unique()
  for date in unique_dates:
    #print(f'running date {date}')
    yield torch.tensor(signals.loc[dates == date].values), torch.tensor(returns.loc[dates == date].values)
    
class FlexibleMLP(nn.Module):
    def __init__(self, layers: list, scale: float=1.):
        """
        param: layers = list of integers
        """
        super(FlexibleMLP, self).__init__()

        self.layers = nn.ModuleList()
        self.activations = nn.ModuleList()

        for i in range(len(layers) - 1):
            layer = nn.Linear(layers[i], layers[i+1])

            # LeCun initialization
            nn.init.normal_(layer.weight, mean=0.0, std=scale * np.sqrt(1 / layers[i]))
            nn.init.normal_(layer.bias, mean=0.0, std=0 * np.sqrt(1 / layers[i]))

            self.layers.append(layer)
            # Add ReLU activation after each layer except the last
            if i < len(layers) - 2:
                self.activations.append(nn.ReLU())
            else:
                # Placeholder for the last layer's activation
                self.activations.append(nn.Identity())

    def forward(self, x, return_last_hidden=False):
        last_hidden = None

        for layer, activation in zip(self.layers[:-1], self.activations[:-1]):
            x = activation(layer(x))
            last_hidden = x  # Update last_hidden at each hidden layer

        # Apply the last layer without ReLU (or Identity for the placeholder)
        x = self.layers[-1](x)

        if return_last_hidden:
            return x, last_hidden
        return x
  
def mssr_loss(output, target):
  """
  MSRR = Maximal Sharpe Ratio Regression
  This is our MSRR loss through which we evaluate the quality of predictions
  Every mini batch is a month. So,
  (output * target.view((output.shape[0], 1))).sum() is the return on the
  portfolio in that particular month (.sum() is over stocks)
  """
  dist = 1 - (output * target.view((output.shape[0], 1))).sum()
  msrr = torch.pow(dist, 2)
  return msrr

def set_seed(seed_value=42):
    """Set seed for reproducibility."""
    np.random.seed(seed_value)  # Set NumPy seed
    torch.manual_seed(seed_value)  # Set PyTorch seed
    random.seed(seed_value)  # Set Python random seed

    # If you are using CUDA:
    if torch.cuda.is_available():
        torch.cuda.manual_seed(seed_value)
        torch.cuda.manual_seed_all(seed_value)  # For multi-GPU
        torch.backends.cudnn.deterministic = True
        torch.backends.cudnn.benchmark = False

# Proceed with the rest of the setup (loss, optimizer) and training loop as before
# Loss and optimizer
ridge_penalty = 0.01  # Regularization strength
set_seed(42)  # Fixing the seed

width = 64
model = FlexibleMLP([signals.shape[1], width, 1], scale=1.) # re-initializing weights !!!
criterion = mssr_loss # this is our custom loss
optimizer = torch.optim.Adam(model.parameters(), lr=0.000001)  # Using Adam optimizer for better performance with deep networks

# Training loop
set_seed(0)  # Fixing the seed
num_epochs = 40  # You might need more epochs for a deep network
for epoch in range(num_epochs):
    for inputs, targets in train_loader(train_signals, train_returns):
        # each mini batch is a month of data
        # Forward pass
        outputs = model(inputs)
        loss = criterion(outputs, targets) # this is (1- portfolio return)^2

        # Backward and optimize
        optimizer.zero_grad()
        loss.backward()
        optimizer.step()

    if (epoch+1) % 2 == 0:
        print(f'Epoch [{epoch+1}/{num_epochs}], Loss: {loss.item():.4f}')

model.eval()
test_data_predictions = model(torch.tensor(test_signals.values))
managed_returns = build_managed_returns(returns=test_returns, signals=pd.DataFrame(test_data_predictions.detach().numpy(), index=test_returns.index))
managed_returns.cumsum().plot()
plt.title(f'sr={sharpe_ratio(managed_returns)}')

class ExpandedMultiHeadAttention(nn.Module):
    def __init__(self, d_input, num_heads):
        super().__init__()
        self.d_input = d_input
        self.num_heads = num_heads
        self.d_proj = d_input * num_heads  # total projection size

        # Linear layers to project to multi-head space
        self.W_q = nn.Linear(d_input, self.d_proj)
        self.W_k = nn.Linear(d_input, self.d_proj)
        self.W_v = nn.Linear(d_input, self.d_proj)

        # Final output layer
        self.output_layer = nn.Linear(self.d_proj, 1)

        self.reset_parameters()

    def reset_parameters(self):
        nn.init.xavier_uniform_(self.W_q.weight)
        nn.init.xavier_uniform_(self.W_k.weight)
        nn.init.xavier_uniform_(self.W_v.weight)
        nn.init.xavier_uniform_(self.output_layer.weight)
        nn.init.zeros_(self.W_q.bias)
        nn.init.zeros_(self.W_k.bias)
        nn.init.zeros_(self.W_v.bias)
        nn.init.zeros_(self.output_layer.bias)

    def forward(self, X, return_attention=False):
        N, d = X.shape
        assert d == self.d_input

        Q = self.W_q(X)  # (N, h*d)
        K = self.W_k(X)
        V = self.W_v(X)

        Q = Q.view(N, self.num_heads, self.d_input)  # (N, h, d)
        K = K.view(N, self.num_heads, self.d_input)
        V = V.view(N, self.num_heads, self.d_input)

        # Compute attention weights (scalar per head)
        Q_ = Q.unsqueeze(2)  # (N, h, 1, d)
        K_ = K.unsqueeze(3)  # (N, h, d, 1)
        attn_scores = torch.matmul(Q_, K_) / (self.d_input ** 0.5)  # (N, h, 1, 1)
        attn_weights = torch.sigmoid(attn_scores).squeeze(-1).squeeze(-1)  # (N, h)

        # Apply attention weights to V
        attn_output = attn_weights.unsqueeze(-1) * V  # (N, h, d)
        attn_output = attn_output.reshape(N, self.d_proj)  # (N, h*d)

        y = self.output_layer(attn_output)  # (N, 1)

        if return_attention:
            return y, attn_weights
        return y
      
def plot_scalar_attention_weights(attn_weights, idx=0, title=None):
    """
    Plot scalar attention weights (shape: N x h) for a specific sample.
    """
    weights = attn_weights[idx].detach().cpu()
    num_heads = weights.shape[0]

    plt.figure(figsize=(num_heads, 3))
    plt.bar(range(num_heads), weights)
    plt.xticks(range(num_heads), [f"Head {i+1}" for i in range(num_heads)])
    plt.ylabel("Attention Weight")
    if title:
        plt.title(title)
    plt.show()
    
d = 6
h = 4
N = 16

model = ExpandedMultiHeadAttention(d_input=d, num_heads=h)
X = torch.randn(N, d)

y_pred, attn_weights = model(X, return_attention=True)
print("Output shape:", y_pred.shape)
print("Attention shape:", attn_weights.shape)

plot_scalar_attention_weights(attn_weights, idx=0, title="Attention Weights (Sample 0)")

N, d = 32, 6    # small, indivisible d
h = 5           # 5 heads, total projection size = 30
X = torch.randn(N, d)

mha = ExpandedMultiHeadAttention(d_input=d, num_heads=h)
y_pred = mha(X)   # shape: (N, 1)
print(y_pred.shape)

# Proceed with the rest of the setup (loss, optimizer) and training loop as before
# Loss and optimizer
set_seed(42)  # Fixing the seed
model = ExpandedMultiHeadAttention(d_input=signals.shape[1], num_heads=1) # re-initializing weights !!!
criterion = mssr_loss # this is our custom loss
optimizer = torch.optim.Adam(model.parameters(), lr=0.00001)  # Using Adam optimizer for better performance with deep networks

# Training loop
set_seed(0)  # Fixing the seed
num_epochs = 40  # You might need more epochs for a deep network
for epoch in range(num_epochs):
    for inputs, targets in train_loader(train_signals, train_returns):
        # each mini batch is a month of data
        # Forward pass
        outputs = model(inputs)
        loss = criterion(outputs, targets) # this is (1- portfolio return)^2

        # Backward and optimize
        optimizer.zero_grad()
        loss.backward()
        optimizer.step()

    if (epoch+1) % 2 == 0:
        print(f'Epoch [{epoch+1}/{num_epochs}], Loss: {loss.item():.4f}')
        
class ExpandedMultiHeadAttention(nn.Module):
    """
    Multi-Head Attention across N samples (no sequence dimension).
    This module computes attention weights between all samples in the batch,
    similar to standard self-attention across tokens in a sequence.

    Each head sees all N samples as both queries and keys.
    """

    def __init__(self, d_input, num_heads, init_fn_dict=None, scale_dict=None):
        """
        Parameters:
        - d_input (int): Input feature dimension.
        - num_heads (int): Number of attention heads.
        - init_fn_dict (dict): Optional dict mapping parameter names to init functions.
        - scale_dict (dict): Optional dict mapping parameter names to scale factors.
        """
        super().__init__()

        self.d_input = d_input                   # Original input dimension
        self.num_heads = num_heads               # Number of attention heads
        self.d_proj = d_input * num_heads        # Projected dimension after multi-head concat

        # Linear projections for Q, K, V. These will be reshaped into heads later
        self.W_q = nn.Linear(d_input, self.d_proj)
        self.W_k = nn.Linear(d_input, self.d_proj)
        self.W_v = nn.Linear(d_input, self.d_proj)

        # Final output layer to map multi-head output to scalar prediction
        self.output_layer = nn.Linear(self.d_proj, 1)

        # Optional dictionaries for custom parameter initialization and scaling
        self.init_fn_dict = init_fn_dict or {}
        self.scale_dict = scale_dict or {}

        # Initialize weights
        self.reset_parameters()

    def reset_parameters(self):
        """
        Applies parameter-specific initialization and scaling.
        Falls back to Xavier initialization and 1.0 scale if unspecified.
        """

        def apply_init(name, param):
            # Use user-specified init function or default to Xavier
            init_fn = self.init_fn_dict.get(name, nn.init.xavier_uniform_)
            init_fn(param)

        def apply_scale(name, param):
            # Multiply weights by a custom scalar (if provided)
            scale = self.scale_dict.get(name, 1.0)
            param.data.mul_(scale)

        # Initialize weights and biases for each projection layer
        for name, layer in zip(['W_q', 'W_k', 'W_v', 'output'],
                               [self.W_q, self.W_k, self.W_v, self.output_layer]):
            apply_init(name + '_weight', layer.weight)  # Initialize weights
            nn.init.zeros_(layer.bias)                  # Zero the bias
            apply_scale(name + '_weight', layer.weight) # Scale the weights

    def forward(self, X, return_attention=False):
        """
        Forward pass of multi-head self-attention.

        Parameters:
        - X: (N, d_input) input tensor
        - return_attention (bool): if True, also return attention weights

        Returns:
        - y: (N, 1) output predictions
        - attn_weights (optional): (N, num_heads, N) attention scores for each head
        """
        N, d = X.shape
        assert d == self.d_input, "Input feature size mismatch"

        # Project input into Q, K, V for all heads: shape (N, h * d_input)
        Q = self.W_q(X)
        K = self.W_k(X)
        V = self.W_v(X)

        # Reshape each into (N, h, d_input) so each head has its own d_input-dim space
        Q = Q.view(N, self.num_heads, self.d_input)
        K = K.view(N, self.num_heads, self.d_input)
        V = V.view(N, self.num_heads, self.d_input)

        # Transpose to (h, N, d_input): each head is a separate batch
        Q = Q.permute(1, 0, 2)
        K = K.permute(1, 0, 2)
        V = V.permute(1, 0, 2)

        # Compute raw attention scores with scaled dot product:
        #   (h, N, d) x (h, d, N) -> (h, N, N)
        attn_scores = torch.matmul(Q, K.transpose(-2, -1)) / (self.d_input ** 0.5)

        # Apply softmax over keys (last dimension): (h, N, N)
        attn_weights = F.softmax(attn_scores, dim=-1)

        # Apply attention weights to values:
        #   (h, N, N) x (h, N, d) -> (h, N, d)
        attn_output = torch.matmul(attn_weights, V)

        # Recombine heads: (h, N, d) -> (N, h, d)
        attn_output = attn_output.permute(1, 0, 2)

        # Flatten across heads: (N, h, d) -> (N, h * d)
        attn_output = attn_output.reshape(N, self.d_proj)

        # Final output projection: (N, h * d) -> (N, 1)
        y = self.output_layer(attn_output)

        # Return predictions and optionally attention weights (transpose to (N, h, N))
        return (y, attn_weights.permute(1, 0, 2)) if return_attention else y

def plot_attention_heatmaps(attn_weights, title=None, idx_range=None):
    """
    Plot attention matrices for each head.

    Parameters:
    - attn_weights: (N, num_heads, N) tensor of attention scores
    - title: optional title for the full figure
    - idx_range: tuple (start, end), optional crop of matrix
    """
    N, num_heads, _ = attn_weights.shape

    # Optionally crop attention matrix to a slice of samples
    if idx_range:
        start, end = idx_range
        attn_weights = attn_weights[start:end, :, start:end]
        N = end - start

    # Create one subplot per head
    fig, axes = plt.subplots(1, num_heads, figsize=(4 * num_heads, 4))
    if num_heads == 1:
        axes = [axes]  # handle single-head case

    for h in range(num_heads):
        ax = axes[h]
        # Show attention weight heatmap: rows = queries, cols = keys
        im = ax.imshow(attn_weights[:, h, :].detach().cpu(), cmap="viridis", aspect='auto')
        ax.set_title(f"Head {h + 1}")
        ax.set_xlabel("Key Index")
        ax.set_ylabel("Query Index")
        fig.colorbar(im, ax=ax, fraction=0.046, pad=0.04)

    if title:
        plt.suptitle(title)
    plt.tight_layout()
    plt.show()

d = 6
h = 4
N = 16

model = ExpandedMultiHeadAttention(d_input=d, num_heads=h)
X = torch.randn(N, d)

y_pred, attn_weights = model(X, return_attention=True)  # attn_weights: (N, h, N)
plot_attention_heatmaps(attn_weights, title="Full N x N Attention per Head")

# Model config
d = 6         # Input dimension
h = 4         # Number of heads
N = 16        # Batch size

# Optional: custom initialization and scaling
init_fns = {
  'W_q_weight': nn.init.kaiming_uniform_,
  'W_k_weight': nn.init.xavier_normal_,
  'W_v_weight': lambda w: nn.init.constant_(w, 0.5),
  'output_weight': nn.init.xavier_uniform_,
}

scales = {
  'W_q_weight': 0.9,
  'W_v_weight': 0.1
}

# Create model
model = ExpandedMultiHeadAttention(d_input=d, num_heads=h,
                                  init_fn_dict=init_fns,
                                  scale_dict=scales)

# Dummy input
X = torch.randn(N, d)

# Forward pass
y_pred, attn_weights = model(X, return_attention=True)
print("Output shape:", y_pred.shape)        # (N, 1)
print("Attention shape:", attn_weights.shape)  # (N, h)
plot_attention_heatmaps(attn_weights, title="Full N x N Attention per Head")

# Optional: custom initialization and scaling
init_fns = {
  'W_q_weight': nn.init.kaiming_uniform_,
  'W_k_weight': nn.init.xavier_normal_,
  'W_v_weight': lambda w: nn.init.constant_(w, 0.),
  'output_weight': nn.init.xavier_uniform_,
}

scales = {
  'W_q_weight': 0.05,
  'W_v_weight': 0.01,
  'W_k_weight': 0.01,
  'output_weight': 10.
}

#heads
h = 1

# Create model
model = ExpandedMultiHeadAttention(d_input=signals.shape[1],
                                   num_heads=h,
                                  init_fn_dict=init_fns,
                                  scale_dict=scales)
criterion = mssr_loss # this is our custom loss
optimizer = torch.optim.Adam(model.parameters(), lr=0.001)  # Using Adam optimizer for better performance with deep networks

# Training loop
set_seed(10)  # Fixing the seed
num_epochs = 60  # You might need more epochs for a deep network
for epoch in range(num_epochs):
    for inputs, targets in train_loader(train_signals, train_returns):
        # each mini batch is a month of data
        # Forward pass
        outputs = model(inputs)
        loss = criterion(outputs, targets) # this is (1- portfolio return)^2

        # Backward and optimize
        optimizer.zero_grad()
        loss.backward()
        optimizer.step()

    if (epoch+1) % 2 == 0:
        print(f'Epoch [{epoch+1}/{num_epochs}], Loss: {loss.item():.4f}')

from tqdm import tqdm 
model.eval()
BATCH_SIZE = 2048
test_signals_np = test_signals.values.astype(np.float32)

all_predictions = []

num_batches = (len(test_signals_np) - 1) // BATCH_SIZE + 1
for i in tqdm(range(num_batches)):
    start_idx = i * BATCH_SIZE
    end_idx = min((i + 1) * BATCH_SIZE, len(test_signals_np))
    batch_data = torch.tensor(test_signals_np[start_idx:end_idx], dtype=torch.float32)
    
    with torch.no_grad():
        batch_pred = model(batch_data)
    
    all_predictions.append(batch_pred.cpu().detach().numpy())

test_data_predictions = np.concatenate(all_predictions, axis=0)

managed_returns = build_managed_returns(returns=test_returns, signals=pd.DataFrame(test_data_predictions, index=test_returns.index))
managed_returns.cumsum().plot()
plt.title(f'sr={sharpe_ratio(managed_returns)}')
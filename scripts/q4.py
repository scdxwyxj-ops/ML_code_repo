
import pandas as pd
import numpy as np
import os
from sklearn.preprocessing import StandardScaler
# ML models
from sklearn.linear_model import Ridge
from sklearn.ensemble import RandomForestRegressor
from xgboost import XGBRegressor
from sklearn.neural_network import MLPRegressor

# Model selection
from sklearn.model_selection import GridSearchCV, TimeSeriesSplit
from sklearn.metrics import mean_squared_error, mean_absolute_error

# Visualization
import matplotlib.pyplot as plt
import seaborn as sns

# Utilities
import joblib
from datetime import datetime

from scipy.optimize import minimize

import warnings
warnings.filterwarnings('ignore')


"""
Data Preparation
====================================================
- Load dataset
- Handle missing values
- Split into train/validation/test sets
"""


#==============================================================================
# Load Data
#==============================================================================
# Data excerpted from Kaggle
# https://www.kaggle.com/datasets/borismarjanovic/price-volume-data-for-all-us-stocks-etfs
# Load CSV file
df = pd.read_csv('stock.csv')

# Convert date column to datetime and set as index
df['Date'] = pd.to_datetime(df['Date'])
df = df.set_index('Date')
df = df.sort_index() 

# Extract stock tickers (columns ending with .us)
stocks = [col for col in df.columns if col.endswith('.us')]
n_stocks = len(stocks)

#==============================================================================
# Handle Missing Values
#==============================================================================

# Check for missing values
missing_counts = df[stocks].isnull().sum()
total_missing = missing_counts.sum()

if total_missing > 0:

    # Forward fill missing values (appropriate for time series price data)
    df[stocks] = df[stocks].ffill()
    
    # Backward fill for any remaining NaN at the beginning
    df[stocks] = df[stocks].bfill()
    
    # Verify no missing values remain
    remaining_missing = df[stocks].isnull().sum().sum()

# Check for non-positive prices (data quality issue)
non_positive_found = False
for stock in stocks:
    non_positive_count = (df[stock] <= 0).sum()
    if non_positive_count > 0:
        # Replace non-positive values with NaN and interpolate
        df.loc[df[stock] <= 0, stock] = np.nan
        df[stock] = df[stock].interpolate(method='linear')


#==============================================================================
# Train/Validation/Test Split (70/15/15)
#==============================================================================

# Time series split: 70% train, 15% validation, 15% test
train_size = int(len(df) * 0.70)
val_size = int(len(df) * 0.15)

train_data = df.iloc[:train_size].copy()
val_data = df.iloc[train_size:train_size + val_size].copy()
test_data = df.iloc[train_size + val_size:].copy()

#==============================================================================
# Save Processed Data
#==============================================================================

# Create output directory
output_dir = 'data_processed'
os.makedirs(output_dir, exist_ok=True)

# Save cleaned full dataset
df.to_csv(f'{output_dir}/cleaned_data.csv')

# Save split datasets
train_data.to_csv(f'{output_dir}/train_data.csv')
val_data.to_csv(f'{output_dir}/val_data.csv')
test_data.to_csv(f'{output_dir}/test_data.csv')

# Save stock list for later use
stock_info = pd.DataFrame({'stock': stocks})
stock_info.to_csv(f'{output_dir}/stock_list.csv', index=False)

"""
Feature Engineering & Label Construction
====================================================
- Construct target variable (future realized volatility)
- Engineer features (price, volatility, technical indicators)
- Feature standardization
- Save final datasets for modeling
"""

#==============================================================================
# Configuration
#==============================================================================

# Prediction horizon
PREDICTION_WINDOW = 20  # Predict volatility over next 20 days (~1 month)

# Feature windows 
SHORT_WINDOW = 20   # Short-term features (1 month)
MEDIUM_WINDOW = 60  # Medium-term features (3 months)
LONG_WINDOW = 120   # Long-term features (6 months)

# Directories
INPUT_DIR = 'data_processed'
OUTPUT_DIR = 'data_features'
os.makedirs(OUTPUT_DIR, exist_ok=True)

# Trading days per year (for annualization)
TRADING_DAYS = 252

#==============================================================================
# Load Data
#==============================================================================

train_data = pd.read_csv(f'{INPUT_DIR}/train_data.csv', index_col='Date', parse_dates=True)
val_data = pd.read_csv(f'{INPUT_DIR}/val_data.csv', index_col='Date', parse_dates=True)
test_data = pd.read_csv(f'{INPUT_DIR}/test_data.csv', index_col='Date', parse_dates=True)

# Combine for feature engineering (will split again later)
df = pd.concat([train_data, val_data, test_data])

# Get stock list
stocks = [col for col in df.columns if col.endswith('.us')]
n_stocks = len(stocks)

#==============================================================================
# Calculate Returns
#==============================================================================

returns = df[stocks].pct_change()

#==============================================================================
# Construct Target Variable (Future Realized Volatility) - FIXED
#==============================================================================

# Calculate future realized volatility for each stock
target_dict = {}

for stock in stocks:

    rolling_vol = returns[stock].rolling(window=PREDICTION_WINDOW).std()
    
    future_vol = rolling_vol.shift(-PREDICTION_WINDOW)
    
    # Annualize
    future_vol = future_vol * np.sqrt(TRADING_DAYS)
    
    target_dict[stock] = future_vol

# Create target dataframe
targets = pd.DataFrame(target_dict, index=df.index)

#==============================================================================
# Feature Engineering
#==============================================================================

feature_dfs = []

# Price-based features

for stock in stocks:
    stock_features = pd.DataFrame(index=df.index)
    
    # Returns at different horizons
    stock_features[f'{stock}_return_1d'] = returns[stock]
    stock_features[f'{stock}_return_5d'] = df[stock].pct_change(5)
    stock_features[f'{stock}_return_20d'] = df[stock].pct_change(20)
    
    # Log returns
    stock_features[f'{stock}_log_return_1d'] = np.log(df[stock] / df[stock].shift(1))
    
    # Moving averages
    stock_features[f'{stock}_ma_20'] = df[stock].rolling(window=SHORT_WINDOW).mean()
    stock_features[f'{stock}_ma_60'] = df[stock].rolling(window=MEDIUM_WINDOW).mean()
    
    # Price relative to moving average
    stock_features[f'{stock}_price_to_ma20'] = df[stock] / stock_features[f'{stock}_ma_20']
    stock_features[f'{stock}_price_to_ma60'] = df[stock] / stock_features[f'{stock}_ma_60']
    
    feature_dfs.append(stock_features)


# Volatility features (Historical & EWMA)

for stock in stocks:
    stock_features = pd.DataFrame(index=df.index)
    
    # Historical Volatility (different windows)
    hist_vol_20 = returns[stock].rolling(window=SHORT_WINDOW).std() * np.sqrt(TRADING_DAYS)
    hist_vol_60 = returns[stock].rolling(window=MEDIUM_WINDOW).std() * np.sqrt(TRADING_DAYS)
    
    stock_features[f'{stock}_hist_vol_20'] = hist_vol_20
    stock_features[f'{stock}_hist_vol_60'] = hist_vol_60
    
    # EWMA Volatility (Exponentially Weighted Moving Average)
    ewma_vol = returns[stock].ewm(span=20, adjust=False).std() * np.sqrt(TRADING_DAYS)
    stock_features[f'{stock}_ewma_vol_20'] = ewma_vol
    
    # Volatility of volatility (risk of risk)
    vol_of_vol = hist_vol_20.rolling(window=SHORT_WINDOW).std()
    stock_features[f'{stock}_vol_of_vol'] = vol_of_vol
    
    feature_dfs.append(stock_features)

# Technical Indicators
for stock in stocks:
    stock_features = pd.DataFrame(index=df.index)
    
    # RSI (Relative Strength Index)
    delta = returns[stock]
    gain = (delta.where(delta > 0, 0)).rolling(window=14).mean()
    loss = (-delta.where(delta < 0, 0)).rolling(window=14).mean()
    rs = gain / loss
    rsi = 100 - (100 / (1 + rs))
    stock_features[f'{stock}_rsi_14'] = rsi
    
    # Bollinger Bands
    ma_20 = df[stock].rolling(window=20).mean()
    std_20 = df[stock].rolling(window=20).std()
    upper_band = ma_20 + (std_20 * 2)
    lower_band = ma_20 - (std_20 * 2)
    stock_features[f'{stock}_bb_position'] = (df[stock] - lower_band) / (upper_band - lower_band)
    
    # MACD (Moving Average Convergence Divergence)
    ema_12 = df[stock].ewm(span=12, adjust=False).mean()
    ema_26 = df[stock].ewm(span=26, adjust=False).mean()
    macd = ema_12 - ema_26
    signal = macd.ewm(span=9, adjust=False).mean()
    stock_features[f'{stock}_macd'] = macd
    stock_features[f'{stock}_macd_signal'] = signal
    stock_features[f'{stock}_macd_diff'] = macd - signal
    
    # Momentum
    stock_features[f'{stock}_momentum_10'] = df[stock] - df[stock].shift(10)
    stock_features[f'{stock}_momentum_20'] = df[stock] - df[stock].shift(20)
    
    feature_dfs.append(stock_features)

# Cross-sectional features (comparing across stocks)

cross_features = pd.DataFrame(index=df.index)

# Market average return
cross_features['market_return_1d'] = returns[stocks].mean(axis=1)
cross_features['market_return_5d'] = df[stocks].pct_change(5).mean(axis=1)

# Market volatility
market_vol = returns[stocks].std(axis=1) * np.sqrt(TRADING_DAYS)
cross_features['market_vol'] = market_vol

# For each stock, calculate relative strength
for stock in stocks:
    # Stock return relative to market
    cross_features[f'{stock}_rel_return'] = returns[stock] - cross_features['market_return_1d']
    
    # Stock volatility relative to market
    stock_vol = returns[stock].rolling(window=SHORT_WINDOW).std() * np.sqrt(TRADING_DAYS)
    cross_features[f'{stock}_rel_vol'] = stock_vol - cross_features['market_vol'].rolling(window=SHORT_WINDOW).mean()
    
    # Rank-based features (percentile across stocks)
    rank_return = returns[stocks].rank(axis=1, pct=True)[stock]
    cross_features[f'{stock}_rank_return'] = rank_return

feature_dfs.append(cross_features)

# Lagged features
for stock in stocks:
    stock_features = pd.DataFrame(index=df.index)
    
    # Lagged returns
    for lag in [1, 2, 5]:
        stock_features[f'{stock}_return_lag{lag}'] = returns[stock].shift(lag)
    
    # Lagged volatility
    hist_vol = returns[stock].rolling(window=SHORT_WINDOW).std() * np.sqrt(TRADING_DAYS)
    for lag in [1, 5]:
        stock_features[f'{stock}_vol_lag{lag}'] = hist_vol.shift(lag)
    
    feature_dfs.append(stock_features)

#==============================================================================
# Create Dataset (Wide Format - Keep Date Index)
#==============================================================================

# Concatenate all feature dataframes
features_df = pd.concat(feature_dfs, axis=1)

# Create feature set
final_features = features_df.copy()

for stock in stocks:
    final_features[f'target_{stock}'] = targets[stock]

# Remove rows where we don't have enough historical data OR future target
feature_cols = [col for col in features_df.columns]
target_cols = [f'target_{stock}' for stock in stocks]

valid_features = final_features[feature_cols].notna().sum(axis=1) > (len(feature_cols) * 0.5)
valid_targets = final_features[target_cols].notna().sum(axis=1) > 0

final_features = final_features[valid_features & valid_targets].copy()

final_features[feature_cols] = final_features[feature_cols].ffill().bfill()

#==============================================================================
# Split Back into Train/Val/Test
#==============================================================================


# Get the original split dates
train_end = train_data.index[-1]
val_end = val_data.index[-1]

# Split
train_final = final_features[final_features.index <= train_end].copy()
val_final = final_features[(final_features.index > train_end) & (final_features.index <= val_end)].copy()
test_final = final_features[final_features.index > val_end].copy()


#==============================================================================
# Feature Standardization
#==============================================================================

# Standardize ONLY feature columns (not target columns)
scaler = StandardScaler()

X_train = train_final[feature_cols]
X_val = val_final[feature_cols]
X_test = test_final[feature_cols]

X_train_scaled = pd.DataFrame(
    scaler.fit_transform(X_train),
    index=X_train.index,
    columns=feature_cols
)

X_val_scaled = pd.DataFrame(
    scaler.transform(X_val),
    index=X_val.index,
    columns=feature_cols
)

X_test_scaled = pd.DataFrame(
    scaler.transform(X_test),
    index=X_test.index,
    columns=feature_cols
)

# Add target columns back
for col in target_cols:
    X_train_scaled[col] = train_final[col]
    X_val_scaled[col] = val_final[col]
    X_test_scaled[col] = test_final[col]


#==============================================================================
# Save Final Datasets
#==============================================================================

# Save complete datasets (features + targets)
X_train_scaled.to_csv(f'{OUTPUT_DIR}/train_final.csv')
X_val_scaled.to_csv(f'{OUTPUT_DIR}/val_final.csv')
X_test_scaled.to_csv(f'{OUTPUT_DIR}/test_final.csv')

# Save feature names and target names
feature_info = pd.DataFrame({
    'feature': feature_cols,
    'type': 'feature'
})
target_info = pd.DataFrame({
    'feature': target_cols,
    'type': 'target'
})
column_info = pd.concat([feature_info, target_info])
column_info.to_csv(f'{OUTPUT_DIR}/column_info.csv', index=False)

# Save stock list
pd.DataFrame({'stock': stocks}).to_csv(f'{OUTPUT_DIR}/stock_list.csv', index=False)

# Save scaler
joblib.dump(scaler, f'{OUTPUT_DIR}/scaler.pkl')

"""
ML Model Training and Evaluation
====================================================
- Train 4 ML models (Ridge, Random Forest, XGBoost, MLP)
- Hyperparameter tuning with TimeSeriesSplit
- Baseline comparison with traditional metrics
- Performance evaluation
"""

#==============================================================================
# Configuration
#==============================================================================

INPUT_DIR = 'data_features'
OUTPUT_DIR = 'models'
RESULTS_DIR = 'results'

os.makedirs(OUTPUT_DIR, exist_ok=True)
os.makedirs(RESULTS_DIR, exist_ok=True)

# Random seed for reproducibility
RANDOM_STATE = 42
np.random.seed(RANDOM_STATE)

#==============================================================================
# Load Data
#==============================================================================

train_data = pd.read_csv(f'{INPUT_DIR}/train_final.csv', index_col=0, parse_dates=True)
val_data = pd.read_csv(f'{INPUT_DIR}/val_final.csv', index_col=0, parse_dates=True)
test_data = pd.read_csv(f'{INPUT_DIR}/test_final.csv', index_col=0, parse_dates=True)

# Load column info
column_info = pd.read_csv(f'{INPUT_DIR}/column_info.csv')
feature_cols = column_info[column_info['type'] == 'feature']['feature'].tolist()
target_cols = column_info[column_info['type'] == 'target']['feature'].tolist()

# Load stock list
stocks = pd.read_csv(f'{INPUT_DIR}/stock_list.csv')['stock'].tolist()


#==============================================================================
# Prepare Pooled Dataset
#==============================================================================

# Create pooled dataset: each stock becomes multiple training samples
def create_pooled_dataset(data, stocks, feature_cols, target_cols):
    """
    Create pooled dataset where each stock creates training samples.
    """
    pooled_X = []
    pooled_y = []
    pooled_stock = []
    pooled_date = []
    
    for stock in stocks:
        # Get features for this stock
        stock_prefix = stock + '_'
        stock_feature_cols = [col for col in feature_cols if col.startswith(stock_prefix)]
        
        # Also include market-level features
        market_cols = [col for col in feature_cols if not any(s + '_' in col for s in stocks)]
        
        # Combine stock-specific and market features
        selected_features = stock_feature_cols + market_cols
        
        X_stock = data[selected_features].values
        y_stock = data[f'target_{stock}'].values
        
        pooled_X.append(X_stock)
        pooled_y.append(y_stock)
        pooled_stock.extend([stock] * len(data))
        pooled_date.extend(data.index.tolist())
    
    X = np.vstack(pooled_X)
    y = np.concatenate(pooled_y)
    
    return X, y, pooled_stock, pooled_date

# Create pooled datasets
X_train, y_train, stock_train, date_train = create_pooled_dataset(
    train_data, stocks, feature_cols, target_cols
)

X_val, y_val, stock_val, date_val = create_pooled_dataset(
    val_data, stocks, feature_cols, target_cols
)

X_test, y_test, stock_test, date_test = create_pooled_dataset(
    test_data, stocks, feature_cols, target_cols
)

# Combine train and val for final training (after hyperparameter tuning)
X_train_val = np.vstack([X_train, X_val])
y_train_val = np.concatenate([y_train, y_val])


#==============================================================================
# Define Models and Hyperparameter Grids
#==============================================================================

# Reduced TimeSeriesSplit for faster training
tscv = TimeSeriesSplit(n_splits=2)  # Reduced from 3 to 2

models_config = {
    'Ridge': {
        'model': Ridge(random_state=RANDOM_STATE),
        'params': {
            'alpha': [0.1, 1.0, 10.0],  # Reduced from 4 to 3 values
        }
    },
    
    'RandomForest': {
        'model': RandomForestRegressor(random_state=RANDOM_STATE, n_jobs=-1),
        'params': {
            'n_estimators': [50, 100],      # Reduced: 2 values instead of 3
            'max_depth': [10, 20],          # Reduced: 2 values instead of 3
            'min_samples_split': [5],       # Fixed to 1 value
            'min_samples_leaf': [2]         # Fixed to 1 value
        }
    },
    
    'XGBoost': {
        'model': XGBRegressor(random_state=RANDOM_STATE, n_jobs=-1, tree_method='hist'),  # Added tree_method='hist' for speed
        'params': {
            'n_estimators': [50, 100],      # Reduced: 2 values
            'max_depth': [3, 5],            # Reduced: 2 values
            'learning_rate': [0.1],         # Fixed to 1 value
            'subsample': [0.8]              # Fixed to 1 value
        }
    },
    
    'MLP': {
        'model': MLPRegressor(random_state=RANDOM_STATE, max_iter=300, early_stopping=True),  # Reduced max_iter, added early_stopping
        'params': {
            'hidden_layer_sizes': [(100,), (50, 50)],  # Reduced: 2 architectures
            'activation': ['relu'],                     # Fixed to 1 value
            'alpha': [0.001, 0.01],                    # Reduced: 2 values
            'learning_rate': ['adaptive']              # Fixed to 1 value
        }
    }
}


#==============================================================================
# Train Models with Hyperparameter Tuning
#==============================================================================


trained_models = {}
best_params = {}
cv_scores = {}

for model_name, config in models_config.items():
    
    # Grid search with time series cross-validation
    grid_search = GridSearchCV(
        estimator=config['model'],
        param_grid=config['params'],
        cv=tscv,
        scoring='neg_mean_squared_error',
        n_jobs=-1,
        verbose=0
    )
    
    # Fit on train set
    grid_search.fit(X_train, y_train)
    
    # Store best model and parameters
    trained_models[model_name] = grid_search.best_estimator_
    best_params[model_name] = grid_search.best_params_
    cv_scores[model_name] = -grid_search.best_score_  # Convert back to positive MSE
    
    # Evaluate on validation set
    val_pred = grid_search.best_estimator_.predict(X_val)
    val_mse = mean_squared_error(y_val, val_pred)
    val_mae = mean_absolute_error(y_val, val_pred)
    

#==============================================================================
# Retrain on Train+Val and Evaluate on Test
#==============================================================================

test_results = {}

for model_name, model in trained_models.items():
    
    # Create new model with best parameters
    model_class = type(model)
    final_model = model_class(**best_params[model_name])
    if hasattr(final_model, 'random_state'):
        final_model.random_state = RANDOM_STATE
    if hasattr(final_model, 'n_jobs'):
        final_model.n_jobs = -1
    
    # Retrain on train+val
    final_model.fit(X_train_val, y_train_val)
    
    # Predict on test
    test_pred = final_model.predict(X_test)
    
    # Calculate metrics
    test_mse = mean_squared_error(y_test, test_pred)
    test_mae = mean_absolute_error(y_test, test_pred)
    test_rmse = np.sqrt(test_mse)
    
    test_results[model_name] = {
        'predictions': test_pred,
        'MSE': test_mse,
        'MAE': test_mae,
        'RMSE': test_rmse
    }
    
    
    # Save model
    joblib.dump(final_model, f'{OUTPUT_DIR}/{model_name}_model.pkl')

#==============================================================================
# Baseline Comparison with Traditional Metrics
#==============================================================================

test_data_raw = pd.read_csv('data_processed/test_data.csv', index_col='Date', parse_dates=True)

returns_test_raw = test_data_raw[stocks].pct_change()

baseline_features_all = {}

for stock in stocks:
    hist_vol_20 = returns_test_raw[stock].rolling(window=20).std() * np.sqrt(252)
    hist_vol_60 = returns_test_raw[stock].rolling(window=60).std() * np.sqrt(252)
    ewma_vol_20 = returns_test_raw[stock].ewm(span=20, adjust=False).std() * np.sqrt(252)
    
    baseline_features_all[f'{stock}_hist_vol_20'] = hist_vol_20
    baseline_features_all[f'{stock}_hist_vol_60'] = hist_vol_60
    baseline_features_all[f'{stock}_ewma_vol_20'] = ewma_vol_20

baseline_features_df = pd.DataFrame(baseline_features_all, index=test_data_raw.index)

baseline_features_aligned = baseline_features_df.loc[test_data.index]

baseline_features_aligned = baseline_features_aligned.ffill().bfill()

baseline_results = {}
baseline_predictions = {
    'hist_vol_20': [],
    'hist_vol_60': [],
    'ewma_vol_20': []
}

for stock in stocks:
    hist_vol_20 = baseline_features_aligned[f'{stock}_hist_vol_20'].values
    hist_vol_60 = baseline_features_aligned[f'{stock}_hist_vol_60'].values
    ewma_vol_20 = baseline_features_aligned[f'{stock}_ewma_vol_20'].values
    
    baseline_predictions['hist_vol_20'].append(hist_vol_20)
    baseline_predictions['hist_vol_60'].append(hist_vol_60)
    baseline_predictions['ewma_vol_20'].append(ewma_vol_20)

# Concatenate predictions for all stocks
for method in baseline_predictions.keys():
    baseline_pred = np.concatenate(baseline_predictions[method])
    
    if np.isnan(baseline_pred).any():
        baseline_pred = np.nan_to_num(baseline_pred, nan=np.nanmedian(baseline_pred))
    
    # Calculate metrics
    mse = mean_squared_error(y_test, baseline_pred)
    mae = mean_absolute_error(y_test, baseline_pred)
    rmse = np.sqrt(mse)
    
    baseline_results[method] = {
        'predictions': baseline_pred,
        'MSE': mse,
        'MAE': mae,
        'RMSE': rmse
    }
    

#==============================================================================
# Create Comparison Table
#==============================================================================

# Combine results
comparison_data = []

# ML models
for model_name, results in test_results.items():
    comparison_data.append({
        'Model': model_name,
        'Type': 'ML',
        'MSE': results['MSE'],
        'MAE': results['MAE'],
        'RMSE': results['RMSE']
    })

# Baseline models
baseline_names = {
    'hist_vol_20': 'Historical Vol (20d)',
    'hist_vol_60': 'Historical Vol (60d)',
    'ewma_vol_20': 'EWMA Vol (20d)'
}

for method, results in baseline_results.items():
    comparison_data.append({
        'Model': baseline_names[method],
        'Type': 'Baseline',
        'MSE': results['MSE'],
        'MAE': results['MAE'],
        'RMSE': results['RMSE']
    })

# Create DataFrame
comparison_df = pd.DataFrame(comparison_data)
comparison_df = comparison_df.sort_values('MSE')

# Save comparison table
comparison_df.to_csv(f'{RESULTS_DIR}/model_comparison.csv', index=False)

#==============================================================================
# Visualizations
#==============================================================================

# Model Comparison Bar Chart
fig, axes = plt.subplots(1, 3, figsize=(15, 5))

metrics = ['MSE', 'MAE', 'RMSE']
colors = ['#2E86AB' if t == 'ML' else '#A23B72' for t in comparison_df['Type']]

for idx, metric in enumerate(metrics):
    ax = axes[idx]
    bars = ax.barh(comparison_df['Model'], comparison_df[metric], color=colors)
    ax.set_xlabel(metric, fontsize=12)
    ax.set_title(f'{metric} Comparison (Lower is Better)', fontsize=12, fontweight='bold')
    ax.invert_yaxis()
    
    # Add value labels
    for bar in bars:
        width = bar.get_width()
        ax.text(width, bar.get_y() + bar.get_height()/2, 
                f'{width:.4f}', 
                ha='left', va='center', fontsize=9)

# Add legend
from matplotlib.patches import Patch
legend_elements = [
    Patch(facecolor='#2E86AB', label='ML Models'),
    Patch(facecolor='#A23B72', label='Traditional Baselines')
]
fig.legend(handles=legend_elements, loc='upper center', ncol=2, 
           bbox_to_anchor=(0.5, 0.02), fontsize=10)

plt.tight_layout(rect=[0, 0.05, 1, 1])
plt.savefig(f'{RESULTS_DIR}/model_comparison.png', dpi=300, bbox_inches='tight')

# Prediction vs Actual Scatter Plot (Best ML model)
best_ml_model = comparison_df[comparison_df['Type'] == 'ML'].iloc[0]['Model']
best_predictions = test_results[best_ml_model]['predictions']

fig, axes = plt.subplots(1, 2, figsize=(14, 5))

# Best ML model
ax = axes[0]
ax.scatter(y_test, best_predictions, alpha=0.5, s=20, edgecolors='none')
ax.plot([y_test.min(), y_test.max()], [y_test.min(), y_test.max()], 
        'r--', lw=2, label='Perfect Prediction')
ax.set_xlabel('Actual Volatility', fontsize=12)
ax.set_ylabel('Predicted Volatility', fontsize=12)
ax.set_title(f'{best_ml_model} Predictions', fontsize=12, fontweight='bold')
ax.legend()
ax.grid(True, alpha=0.3)

# Best baseline
ax = axes[1]
best_baseline = comparison_df[comparison_df['Type'] == 'Baseline'].iloc[0]['Model']
best_baseline_key = [k for k, v in baseline_names.items() if v == best_baseline][0]
baseline_pred = baseline_results[best_baseline_key]['predictions']

ax.scatter(y_test, baseline_pred, alpha=0.5, s=20, edgecolors='none', color='#A23B72')
ax.plot([y_test.min(), y_test.max()], [y_test.min(), y_test.max()], 
        'r--', lw=2, label='Perfect Prediction')
ax.set_xlabel('Actual Volatility', fontsize=12)
ax.set_ylabel('Predicted Volatility', fontsize=12)
ax.set_title(f'{best_baseline} Predictions', fontsize=12, fontweight='bold')
ax.legend()
ax.grid(True, alpha=0.3)

plt.tight_layout()
plt.savefig(f'{RESULTS_DIR}/predictions_scatter.png', dpi=300, bbox_inches='tight')

# Residual Analysis
fig, axes = plt.subplots(2, 2, figsize=(14, 10))

models_to_plot = [
    (best_ml_model, test_results[best_ml_model]['predictions'], 'ML'),
    (best_baseline, baseline_pred, 'Baseline')
]

for idx, (model_name, predictions, model_type) in enumerate(models_to_plot):
    residuals = y_test - predictions
    
    # Residual distribution
    ax = axes[idx, 0]
    ax.hist(residuals, bins=50, alpha=0.7, edgecolor='black')
    ax.axvline(x=0, color='r', linestyle='--', linewidth=2)
    ax.set_xlabel('Residual', fontsize=11)
    ax.set_ylabel('Frequency', fontsize=11)
    ax.set_title(f'{model_name} - Residual Distribution', fontsize=11, fontweight='bold')
    ax.text(0.05, 0.95, f'Mean: {residuals.mean():.4f}\nStd: {residuals.std():.4f}',
            transform=ax.transAxes, verticalalignment='top',
            bbox=dict(boxstyle='round', facecolor='wheat', alpha=0.5))
    
    # Residual vs Predicted
    ax = axes[idx, 1]
    ax.scatter(predictions, residuals, alpha=0.5, s=20, edgecolors='none')
    ax.axhline(y=0, color='r', linestyle='--', linewidth=2)
    ax.set_xlabel('Predicted Volatility', fontsize=11)
    ax.set_ylabel('Residual', fontsize=11)
    ax.set_title(f'{model_name} - Residual vs Predicted', fontsize=11, fontweight='bold')
    ax.grid(True, alpha=0.3)

plt.tight_layout()
plt.savefig(f'{RESULTS_DIR}/residual_analysis.png', dpi=300, bbox_inches='tight')

plt.close('all')

#==============================================================================
# Save Detailed Results
#==============================================================================

# Save predictions
predictions_df = pd.DataFrame({
    'date': date_test,
    'stock': stock_test,
    'actual': y_test,
})

for model_name, results in test_results.items():
    predictions_df[f'pred_{model_name}'] = results['predictions']

for method, results in baseline_results.items():
    predictions_df[f'pred_{method}'] = results['predictions']

predictions_df.to_csv(f'{RESULTS_DIR}/all_predictions.csv', index=False)

# Save best parameters
best_params_df = pd.DataFrame([
    {'Model': model_name, 'Parameters': str(params)}
    for model_name, params in best_params.items()
])
best_params_df.to_csv(f'{RESULTS_DIR}/best_hyperparameters.csv', index=False)

"""
Portfolio Construction and Backtesting
====================================================
- Use ALL ML predictions to construct portfolios
- Compare with traditional methods
- Evaluate actual portfolio performance
"""

#==============================================================================
# Configuration
#==============================================================================

RESULTS_DIR = 'results'
PORTFOLIO_DIR = 'portfolio_results'
os.makedirs(PORTFOLIO_DIR, exist_ok=True)

# Risk-free rate (annualized) - use 0 for simplicity
RISK_FREE_RATE = 0.0

# Rebalancing frequency (days)
REBALANCE_FREQ = 20  # Monthly rebalancing

#==============================================================================
# Load Predictions and Actual Data
#==============================================================================

# Load predictions
predictions_df = pd.read_csv(f'{RESULTS_DIR}/all_predictions.csv')
predictions_df['date'] = pd.to_datetime(predictions_df['date'])

# Load test data for returns
test_data_raw = pd.read_csv('data_processed/test_data.csv', index_col='Date', parse_dates=True)
stocks = [col for col in test_data_raw.columns if col.endswith('.us')]


# Calculate returns
returns_test = test_data_raw[stocks].pct_change().dropna()


#==============================================================================
# Helper Functions
#==============================================================================

def construct_covariance_from_volatility(volatilities, correlation_matrix):
    """
    Construct covariance matrix from volatility vector and correlation matrix.
    Σ = D * P * D, where D is diagonal matrix of volatilities, P is correlation
    """
    D = np.diag(volatilities)
    return D @ correlation_matrix @ D

def minimum_variance_optimization(cov_matrix, long_only=True):
    """
    Solve minimum variance portfolio optimization.
    min w^T Σ w
    s.t. sum(w) = 1, w >= 0 (if long_only)
    """
    n = len(cov_matrix)
    
    # Objective function
    def objective(w):
        return w.T @ cov_matrix @ w
    
    # Constraints
    constraints = [
        {'type': 'eq', 'fun': lambda w: np.sum(w) - 1}  # weights sum to 1
    ]
    
    # Bounds
    if long_only:
        bounds = tuple((0, 1) for _ in range(n))
    else:
        bounds = tuple((-1, 1) for _ in range(n))
    
    # Initial guess (equal weight)
    w0 = np.ones(n) / n
    
    # Optimize
    result = minimize(
        objective,
        w0,
        method='SLSQP',
        bounds=bounds,
        constraints=constraints,
        options={'maxiter': 1000}
    )
    
    if not result.success:
        return np.ones(n) / n  # Return equal weight as fallback
    
    return result.x

def calculate_portfolio_metrics(portfolio_returns):
    """Calculate comprehensive portfolio performance metrics."""
    # Annualized return
    total_return = (1 + portfolio_returns).prod() - 1
    n_periods = len(portfolio_returns)
    ann_return = (1 + total_return) ** (252 / n_periods) - 1
    
    # Annualized volatility
    ann_vol = portfolio_returns.std() * np.sqrt(252)
    
    # Sharpe ratio
    sharpe = (ann_return - RISK_FREE_RATE) / ann_vol if ann_vol > 0 else 0
    
    # Maximum drawdown
    cumulative = (1 + portfolio_returns).cumprod()
    running_max = cumulative.expanding().max()
    drawdown = (cumulative - running_max) / running_max
    max_dd = drawdown.min()
    
    # Downside deviation (semi-deviation)
    downside_returns = portfolio_returns[portfolio_returns < 0]
    downside_dev = downside_returns.std() * np.sqrt(252) if len(downside_returns) > 0 else 0
    
    return {
        'Annualized Return': ann_return,
        'Annualized Volatility': ann_vol,
        'Sharpe Ratio': sharpe,
        'Max Drawdown': max_dd,
        'Downside Deviation': downside_dev
    }

def calculate_turnover(weights_history):
    """Calculate portfolio turnover (average absolute weight change)."""
    if len(weights_history) < 2:
        return 0
    
    turnovers = []
    for i in range(1, len(weights_history)):
        turnover = np.abs(weights_history[i] - weights_history[i-1]).sum()
        turnovers.append(turnover)
    
    return np.mean(turnovers) if turnovers else 0

#==============================================================================
# Prepare Historical Correlation Matrix
#==============================================================================

# Use full test period for correlation (could also use expanding window)
correlation_matrix = returns_test.corr().values


#==============================================================================
# Create Volatility Forecasts for Each Rebalancing Date
#==============================================================================

# Get unique dates
unique_dates = sorted(predictions_df['date'].unique())
rebalance_dates = unique_dates[::REBALANCE_FREQ]  # Rebalance every REBALANCE_FREQ days


# For each rebalancing date, get volatility forecasts for all stocks
def get_volatility_forecast(date, method='pred_Ridge'):
    """Get volatility forecast for all stocks at a given date."""
    date_data = predictions_df[predictions_df['date'] == date]
    
    # Ensure we have all stocks
    if len(date_data) != len(stocks):
        return None
    
    # Create a dictionary mapping stock to volatility
    vol_dict = {}
    for _, row in date_data.iterrows():
        vol_dict[row['stock']] = row[method]
    
    # Return volatility vector in same order as stocks list
    return np.array([vol_dict[stock] for stock in stocks])

#==============================================================================
# Define All Strategies (4 ML + 3 Traditional + Equal Weight)
#==============================================================================

strategies = {
    # ML Models
    'ML-Ridge': 'pred_Ridge',
    'ML-RandomForest': 'pred_RandomForest',
    'ML-XGBoost': 'pred_XGBoost',
    'ML-MLP': 'pred_MLP',
    
    # Traditional Baselines
    'Baseline-Hist20d': 'pred_hist_vol_20',
    'Baseline-Hist60d': 'pred_hist_vol_60',
    'Baseline-EWMA': 'pred_ewma_vol_20',
    
    # Naive strategy
    'Equal Weight': None  # Special case
}


#==============================================================================
# Backtest All Strategies
#==============================================================================

portfolio_results = {}
weights_history = {}

for strategy_name, pred_column in strategies.items():
    
    portfolio_returns_list = []
    weights_list = []
    current_weights = None
    
    for i, date in enumerate(unique_dates):
        
        # Check if it's a rebalancing date
        if date in rebalance_dates:
            if strategy_name == 'Equal Weight':
                # Equal weight strategy
                current_weights = np.ones(len(stocks)) / len(stocks)
            else:
                # Get volatility forecast
                vol_forecast = get_volatility_forecast(date, pred_column)
                
                if vol_forecast is None:
                    if current_weights is None:
                        current_weights = np.ones(len(stocks)) / len(stocks)
                else:
                    # Construct covariance matrix
                    cov_matrix = construct_covariance_from_volatility(
                        vol_forecast, 
                        correlation_matrix
                    )
                    
                    # Optimize
                    current_weights = minimum_variance_optimization(cov_matrix)
            
            weights_list.append(current_weights.copy())
        
        # If no weights yet (first day), use equal weight
        if current_weights is None:
            current_weights = np.ones(len(stocks)) / len(stocks)
        
        # Calculate portfolio return for this day
        if date in returns_test.index:
            daily_returns = returns_test.loc[date].values
            portfolio_return = np.dot(current_weights, daily_returns)
            portfolio_returns_list.append(portfolio_return)
    
    # Convert to pandas Series
    portfolio_returns = pd.Series(
        portfolio_returns_list, 
        index=returns_test.index[:len(portfolio_returns_list)]
    )
    
    # Calculate metrics
    metrics = calculate_portfolio_metrics(portfolio_returns)
    
    # Calculate turnover
    turnover = calculate_turnover(weights_list)
    metrics['Turnover'] = turnover
    
    portfolio_results[strategy_name] = {
        'returns': portfolio_returns,
        'weights': weights_list,
        'metrics': metrics
    }


#==============================================================================
# Create Performance Comparison Table
#==============================================================================

comparison_rows = []
for strategy_name, results in portfolio_results.items():
    metrics = results['metrics']
    
    # Classify strategy type
    if strategy_name.startswith('ML-'):
        strategy_type = 'ML'
    elif strategy_name.startswith('Baseline-'):
        strategy_type = 'Baseline'
    else:
        strategy_type = 'Naive'
    
    comparison_rows.append({
        'Strategy': strategy_name,
        'Type': strategy_type,
        'Ann. Return': metrics['Annualized Return'],
        'Ann. Volatility': metrics['Annualized Volatility'],
        'Sharpe Ratio': metrics['Sharpe Ratio'],
        'Max Drawdown': metrics['Max Drawdown'],
        'Downside Dev': metrics['Downside Deviation'],
        'Turnover': metrics['Turnover']
    })

performance_df = pd.DataFrame(comparison_rows)
performance_df = performance_df.sort_values('Sharpe Ratio', ascending=False)

# Save
performance_df.to_csv(f'{PORTFOLIO_DIR}/portfolio_performance.csv', index=False)


#==============================================================================
# Visualizations
#==============================================================================

# Cumulative Returns
fig, ax = plt.subplots(figsize=(14, 7))

colors = {
    'ML-Ridge': '#1f77b4',
    'ML-RandomForest': '#ff7f0e',
    'ML-XGBoost': '#2ca02c',
    'ML-MLP': '#d62728',
    'Baseline-Hist20d': '#9467bd',
    'Baseline-Hist60d': '#8c564b',
    'Baseline-EWMA': '#e377c2',
    'Equal Weight': '#7f7f7f'
}

for strategy_name, results in portfolio_results.items():
    cumulative = (1 + results['returns']).cumprod()
    ax.plot(cumulative.index, cumulative.values, 
            label=strategy_name, linewidth=2, 
            color=colors.get(strategy_name, 'black'))

ax.set_xlabel('Date', fontsize=12)
ax.set_ylabel('Cumulative Return', fontsize=12)
ax.set_title('Cumulative Returns Comparison (All Strategies)', fontsize=14, fontweight='bold')
ax.legend(fontsize=9, loc='best')
ax.grid(True, alpha=0.3)
plt.tight_layout()
plt.savefig(f'{PORTFOLIO_DIR}/cumulative_returns.png', dpi=300, bbox_inches='tight')

# Performance Metrics Bar Chart (Only top strategies)
top_n = 5
top_strategies = performance_df.head(top_n)

fig, axes = plt.subplots(2, 3, figsize=(18, 10))

metrics_to_plot = [
    ('Ann. Return', 'Annualized Return', True),
    ('Ann. Volatility', 'Annualized Volatility', False),
    ('Sharpe Ratio', 'Sharpe Ratio', True),
    ('Max Drawdown', 'Maximum Drawdown', False),
    ('Downside Dev', 'Downside Deviation', False),
    ('Turnover', 'Turnover', False)
]

for idx, (col, title, higher_better) in enumerate(metrics_to_plot):
    ax = axes[idx // 3, idx % 3]
    
    # Color by type
    colors_list = ['#2E86AB' if t == 'ML' else '#A23B72' if t == 'Baseline' else '#F18F01' 
                   for t in top_strategies['Type']]
    
    bars = ax.bar(range(len(top_strategies)), top_strategies[col], color=colors_list)
    
    ax.set_ylabel(col, fontsize=11)
    ax.set_title(f'{title}{"" if higher_better else " (Lower is Better)"}', 
                 fontsize=11, fontweight='bold')
    ax.set_xticks(range(len(top_strategies)))
    ax.set_xticklabels(top_strategies['Strategy'], rotation=45, ha='right')
    
    # Add value labels
    for bar in bars:
        height = bar.get_height()
        ax.text(bar.get_x() + bar.get_width()/2., height,
                f'{height:.4f}',
                ha='center', va='bottom', fontsize=9)

plt.tight_layout()
plt.savefig(f'{PORTFOLIO_DIR}/performance_metrics.png', dpi=300, bbox_inches='tight')

# Sharpe Ratio Comparison (All strategies)
fig, ax = plt.subplots(figsize=(12, 6))

colors_list = ['#2E86AB' if t == 'ML' else '#A23B72' if t == 'Baseline' else '#F18F01' 
               for t in performance_df['Type']]

bars = ax.barh(range(len(performance_df)), performance_df['Sharpe Ratio'], color=colors_list)
ax.set_yticks(range(len(performance_df)))
ax.set_yticklabels(performance_df['Strategy'])
ax.set_xlabel('Sharpe Ratio', fontsize=12)
ax.set_title('Sharpe Ratio Comparison (All Strategies)', fontsize=14, fontweight='bold')
ax.invert_yaxis()

# Add value labels
for i, bar in enumerate(bars):
    width = bar.get_width()
    ax.text(width, bar.get_y() + bar.get_height()/2, 
            f'{width:.4f}',
            ha='left', va='center', fontsize=9)

# Add legend
from matplotlib.patches import Patch
legend_elements = [
    Patch(facecolor='#2E86AB', label='ML Models'),
    Patch(facecolor='#A23B72', label='Traditional Baselines'),
    Patch(facecolor='#F18F01', label='Naive Strategy')
]
ax.legend(handles=legend_elements, loc='lower right', fontsize=10)

plt.tight_layout()
plt.savefig(f'{PORTFOLIO_DIR}/sharpe_comparison.png', dpi=300, bbox_inches='tight')

# Rolling Volatility
fig, ax = plt.subplots(figsize=(14, 6))

window = 60  # 60-day rolling window
for strategy_name, results in portfolio_results.items():
    rolling_vol = results['returns'].rolling(window=window).std() * np.sqrt(252)
    ax.plot(rolling_vol.index, rolling_vol.values, 
            label=strategy_name, linewidth=1.5, alpha=0.8,
            color=colors.get(strategy_name, 'black'))

ax.set_xlabel('Date', fontsize=12)
ax.set_ylabel('Rolling Volatility (60-day, annualized)', fontsize=12)
ax.set_title('Rolling Volatility Comparison', fontsize=14, fontweight='bold')
ax.legend(fontsize=9, loc='best')
ax.grid(True, alpha=0.3)
plt.tight_layout()
plt.savefig(f'{PORTFOLIO_DIR}/rolling_volatility.png', dpi=300, bbox_inches='tight')

# Drawdown Chart (Top 5 strategies)
fig, ax = plt.subplots(figsize=(14, 6))

for strategy_name in top_strategies['Strategy']:
    results = portfolio_results[strategy_name]
    cumulative = (1 + results['returns']).cumprod()
    running_max = cumulative.expanding().max()
    drawdown = (cumulative - running_max) / running_max
    ax.plot(drawdown.index, drawdown.values * 100, 
            label=strategy_name, linewidth=2)

ax.set_xlabel('Date', fontsize=12)
ax.set_ylabel('Drawdown (%)', fontsize=12)
ax.set_title('Drawdown Comparison (Top 5 Strategies)', fontsize=14, fontweight='bold')
ax.legend(fontsize=10)
ax.grid(True, alpha=0.3)
ax.fill_between(ax.get_xlim(), 0, -100, alpha=0.1, color='red')
plt.tight_layout()
plt.savefig(f'{PORTFOLIO_DIR}/drawdown.png', dpi=300, bbox_inches='tight')

plt.close('all')

#==============================================================================
# Weight Analysis 
#==============================================================================

# Save final weights for selected strategies (ML + Best Baseline)
selected_strategies = ['ML-Ridge', 'ML-XGBoost', 'Baseline-Hist60d', 'Equal Weight']
final_weights = {}

for strategy_name in selected_strategies:
    if strategy_name in portfolio_results and len(portfolio_results[strategy_name]['weights']) > 0:
        final_weights[strategy_name] = portfolio_results[strategy_name]['weights'][-1]

# Create weights comparison DataFrame
if final_weights:
    weights_df = pd.DataFrame(final_weights, index=stocks)
    weights_df.to_csv(f'{PORTFOLIO_DIR}/final_weights.csv')
    
    # Plot weight heatmap
    fig, ax = plt.subplots(figsize=(10, 8))
    im = ax.imshow(weights_df.values, cmap='RdYlBu_r', aspect='auto', vmin=0, vmax=0.3)
    
    ax.set_xticks(np.arange(len(weights_df.columns)))
    ax.set_yticks(np.arange(len(weights_df.index)))
    ax.set_xticklabels(weights_df.columns)
    ax.set_yticklabels(weights_df.index)
    
    plt.setp(ax.get_xticklabels(), rotation=45, ha="right", rotation_mode="anchor")
    
    # Add colorbar
    cbar = plt.colorbar(im, ax=ax)
    cbar.set_label('Weight', rotation=270, labelpad=20)
    
    # Add text annotations
    for i in range(len(stocks)):
        for j in range(len(weights_df.columns)):
            text = ax.text(j, i, f'{weights_df.values[i, j]:.3f}',
                          ha="center", va="center", color="black", fontsize=8)
    
    ax.set_title('Final Portfolio Weights Comparison', fontsize=14, fontweight='bold')
    plt.tight_layout()
    plt.savefig(f'{PORTFOLIO_DIR}/weights_heatmap.png', dpi=300, bbox_inches='tight')
    
    plt.close('all')

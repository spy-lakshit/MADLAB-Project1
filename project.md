Project 2: Algorithmic Market Regime Identification & Trend Prediction
(Clustering + Regression Hybrid System)

Problem Statement: Financial markets operate in different "regimes" (e.g., high volatility, slow uptrends, sideways consolidation). Applying a single regression model to predict price changes across all days results in poor accuracy because market dynamics shift constantly.

Objective: To design a hybrid trading analysis tool that first clusters historical market data into distinct regimes without human labeling, and then applies localized regression to predict short-term price movements specifically tailored to that regime.

Algorithms Used & Justification:

K-Means Clustering: An unsupervised algorithm used to group trading days into regimes (e.g., Cluster 0 = Quiet, Cluster 1 = Volatile) based on volume and price variance.

Locally Weighted Regression (LWR): Standard linear regression assumes a global straight-line relationship. Market data is highly non-linear. LWR gives more weight to historical data points that are conceptually closer to today's market conditions, making it excellent for noisy financial data.

Dataset Structure (Sample CSV):

Daily_Return_Pct (numeric)

Intraday_Volatility (numeric)

Volume_Change_Pct (numeric)

Next_Day_Return (numeric, Target for Regression)

Complete Python Code:

Python
import numpy as np
import pandas as pd
from sklearn.cluster import KMeans
from sklearn.metrics import mean_squared_error
import matplotlib.pyplot as plt

# 1. Generate Synthetic Market Data
np.random.seed(42)
n_days = 500
data = {
    'Daily_Return': np.random.normal(0, 1.5, n_days),
    'Volatility': np.random.uniform(0.5, 3.0, n_days),
    'Volume_Change': np.random.normal(0, 5, n_days),
    'Next_Day_Return': np.random.normal(0.1, 1.0, n_days) # Target
}
df = pd.DataFrame(data)

# Features for clustering (Market conditions)
X_features = df[['Daily_Return', 'Volatility', 'Volume_Change']].values
y_target = df['Next_Day_Return'].values

# 2. Apply K-Means to identify Market Regimes
kmeans = KMeans(n_clusters=3, random_state=42, n_init=10)
df['Regime'] = kmeans.fit_predict(X_features)
print(f"Market Regimes Identified: \n{df['Regime'].value_counts()}\n")

# 3. Custom Locally Weighted Regression (LWR) function
def locally_weighted_regression(x_query, X_train, y_train, tau=0.5):
    # Calculate diagonal weight matrix based on Euclidean distance
    distances = np.sum((X_train - x_query)**2, axis=1)
    weights = np.exp(-distances / (2 * tau**2))
    W = np.diag(weights)
    
    # Add bias term (column of 1s) to X_train and x_query
    X_train_bias = np.c_[np.ones(len(X_train)), X_train]
    x_query_bias = np.r_[1, x_query]
    
    # Calculate theta: (X^T * W * X)^-1 * X^T * W * y
    theta = np.linalg.inv(X_train_bias.T @ W @ X_train_bias) @ (X_train_bias.T @ W @ y_train)
    
    # Predict
    return x_query_bias @ theta

# 4. Hybrid Prediction: Predict a new day based ONLY on its regime
new_day_conditions = np.array([-0.5, 1.2, 2.0]) # A new theoretical day

# Step A: Find which regime this new day belongs to
assigned_regime = kmeans.predict(new_day_conditions.reshape(1, -3))[0]

# Step B: Filter historical data to only include days from the same regime
regime_data = df[df['Regime'] == assigned_regime]
X_regime_train = regime_data[['Daily_Return', 'Volatility', 'Volume_Change']].values
y_regime_train = regime_data['Next_Day_Return'].values

# Step C: Apply LWR using only regime-specific data
prediction = locally_weighted_regression(new_day_conditions, X_regime_train, y_regime_train, tau=1.0)

print(f"New Day Conditions: {new_day_conditions}")
print(f"Assigned Market Regime: Cluster {assigned_regime}")
print(f"Predicted Next Day Return (using LWR): {prediction:.3f}%")
Step-by-Step Explanation:

Clustering: K-Means groups the 500 historical trading days into 3 clusters (regimes) by minimizing the variance within each group.

LWR Implementation: We define a custom mathematical function for LWR. It calculates an exponential weight matrix W. Data points closer to our query point receive higher weights.

Hybrid Pipeline: When a new day's data comes in, the system first asks K-Means "what type of market is this?". It filters history for similar days, then runs LWR to predict the exact price movement based heavily on the most structurally similar days within that regime.

Output Explanation & Metrics:
The output shows the distribution of market regimes (clusters). For evaluation, you would calculate the Mean Squared Error (MSE) or Mean Absolute Error (MAE) between the LWR predictions and the actual Next_Day_Return.

Advantages & Limitations:

Advantage: Highly adaptive. Unlike global linear regression, LWR combined with clustering ensures predictions are hyper-localized to current market behavior.

Limitation: LWR is a "lazy learning" algorithm. It doesn't build a static model; it must recalculate the weights and matrix inversion for every single prediction, making it computationally expensive for massive datasets.

Future Scope:
Integrating real-time API data to feed live market parameters into the model, automating the trigger of a trade if the LWR prediction crosses a specific profitability threshold.

Viva Questions & Answers:

Q: How did you choose the 'k' value (number of clusters) in K-Means?

A: In practice, I would use the Elbow Method by plotting the Within-Cluster Sum of Squares (WCSS) against different values of k. Here, k=3 was chosen conceptually to represent bullish, bearish, and sideways market regimes.

Q: How does Locally Weighted Regression differ from standard Linear Regression?

A: Standard linear regression computes one set of parameters (theta) for the entire dataset to minimize the global cost function. LWR computes a new set of parameters for each prediction by giving a higher weight to training points that are spatially closer to the query point, using the bandwidth parameter (tau).

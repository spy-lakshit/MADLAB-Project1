# Algorithmic Market Regime & Trend Predictor 📈🤖

A hybrid machine learning pipeline that identifies hidden financial market regimes using unsupervised clustering, and predicts short-term price action using highly localized regression.

## 🚀 Overview
Financial markets are highly non-linear and constantly shift between different "regimes" (e.g., low-volatility uptrends vs. high-volatility crashes). Applying standard global regression models to this data results in poor accuracy. 

This system solves this by:
1. Using **K-Means Clustering** to identify the current market regime based on volume and volatility.
2. Filtering historical data to match the current regime.
3. Applying **Locally Weighted Regression (LWR)** to predict the next day's return, mathematically weighting recent and structurally similar days heavier than distant data.

## 🛠️ Tech Stack
* **Language:** Python 3.x
* **Data Processing:** Pandas, NumPy
* **Machine Learning:** Scikit-Learn (K-Means), Custom mathematical implementation for LWR.
* **Frontend Demo:** HTML5, Bootstrap 5

## 📊 Dataset Features
The model requires the following daily parameters:
* `Daily_Return_Pct` (Percentage change from previous close)
* `Intraday_Volatility` (High/Low spread)
* `Volume_Change_Pct` (Percentage change in trading volume)
* `Target:` `Next_Day_Return`

## ⚙️ Installation & Usage
1. Clone the repository:
   ```bash
   git clone [https://github.com/lakshit-v/market-regime-predictor.git](https://github.com/lakshit-v/market-regime-predictor.git)
   ```
2. Install dependencies:
   pip install -r requirements.txt
   
3. Run the analysis script:
   python market_model.py

🧠 Why Locally Weighted Regression (LWR)?
Unlike standard Linear Regression which calculates a single line of best fit for the entire dataset, LWR is a non-parametric, lazy-learning algorithm. It computes a new, localized model for every single prediction, making it exceptionally well-suited for noisy, non-linear financial data.

4. UI Idea: Web Dashboard (HTML/CSS)
This is a clean, "fintech" style dashboard. It allows a user (or examiner) to input current market conditions and visually see both the assigned cluster and the predicted return.

**`index.html`**
```html
<!DOCTYPE html>
<html lang="en">
<head>
    <meta charset="UTF-8">
    <meta name="viewport" content="width=device-width, initial-scale=1.0">
    <title>Market Regime Predictive Engine</title>
    <link href="https://cdn.jsdelivr.net/npm/bootstrap@5.3.0/dist/css/bootstrap.min.css" rel="stylesheet">
    <style>
        body { background-color: #0f172a; color: #f8fafc; font-family: 'Segoe UI', Tahoma, Geneva, Verdana, sans-serif; }
        .card { background-color: #1e293b; border: 1px solid #334155; border-radius: 12px; box-shadow: 0 4px 6px rgba(0,0,0,0.3); }
        .form-label { color: #94a3b8; font-weight: 500; }
        .form-control { background-color: #0f172a; border: 1px solid #475569; color: #f8fafc; }
        .form-control:focus { background-color: #0f172a; color: #f8fafc; border-color: #3b82f6; box-shadow: 0 0 0 0.25rem rgba(59, 130, 246, 0.25); }
        .btn-primary { background-color: #3b82f6; border: none; font-weight: 600; padding: 10px; }
        .btn-primary:hover { background-color: #2563eb; }
        .data-display { background-color: #0f172a; padding: 15px; border-radius: 8px; border: 1px solid #334155; margin-bottom: 15px; }
        .text-bullish { color: #10b981; }
        .text-bearish { color: #ef4444; }
        .text-neutral { color: #eab308; }
    </style>
</head>
<body>

<div class="container mt-5">
    <h2 class="text-center mb-1">Algorithmic Market Prediction Engine</h2>
    <p class="text-center text-secondary mb-5">Hybrid K-Means & Locally Weighted Regression Model</p>
    
    <div class="row justify-content-center">
        <div class="col-md-5 mb-4">
            <div class="card p-4 h-100">
                <h5 class="mb-4" style="color: #e2e8f0;">Input Daily Market Metrics</h5>
                <form id="marketForm">
                    <div class="mb-3">
                        <label class="form-label">Daily Return (%)</label>
                        <input type="number" step="0.01" class="form-control" id="dailyReturn" placeholder="e.g. -1.2" required>
                    </div>
                    <div class="mb-3">
                        <label class="form-label">Intraday Volatility Index</label>
                        <input type="number" step="0.01" class="form-control" id="volatility" placeholder="e.g. 2.5" required>
                    </div>
                    <div class="mb-4">
                        <label class="form-label">Volume Change (%)</label>
                        <input type="number" step="0.1" class="form-control" id="volume" placeholder="e.g. 15.0" required>
                    </div>
                    <button type="button" class="btn btn-primary w-100" onclick="runAnalysis()">Execute Analysis</button>
                </form>
            </div>
        </div>
        
        <div class="col-md-5 mb-4">
            <div class="card p-4 h-100 d-flex flex-column">
                <h5 class="mb-4" style="color: #e2e8f0;">Model Output</h5>
                
                <div class="data-display flex-grow-1 d-flex flex-column justify-content-center text-center">
                    <p class="form-label text-uppercase mb-1" style="font-size: 0.85rem;">Detected Market Regime (K-Means)</p>
                    <h3 id="regimeDisplay" class="text-muted mb-4">--</h3>
                    
                    <p class="form-label text-uppercase mb-1" style="font-size: 0.85rem;">Predicted Next Day Return (LWR)</p>
                    <h1 id="predictionDisplay" class="text-muted">--</h1>
                </div>
                <p class="text-center mt-3 mb-0" style="font-size: 0.8rem; color: #64748b;">Confidence weighting applied via Gaussian Kernel (\(\tau = 1.0\))</p>
            </div>
        </div>
    </div>
</div>

<script>
    // Simulation function for the frontend demo. 
    // In production, this posts data to your Python backend (e.g., Flask/FastAPI).
    function runAnalysis() {
        const ret = parseFloat(document.getElementById('dailyReturn').value);
        const vol = parseFloat(document.getElementById('volatility').value);
        
        const regimeDisplay = document.getElementById('regimeDisplay');
        const predictionDisplay = document.getElementById('predictionDisplay');

        // Mock clustering logic based on volatility
        if (vol > 2.0 && ret < 0) {
            regimeDisplay.innerHTML = "Cluster 2: High-Vol Distribution";
            regimeDisplay.className = "text-bearish";
            predictionDisplay.innerHTML = "-0.85%";
            predictionDisplay.className = "text-bearish";
        } else if (vol <= 1.5 && ret > 0) {
            regimeDisplay.innerHTML = "Cluster 0: Low-Vol Accumulation";
            regimeDisplay.className = "text-bullish";
            predictionDisplay.innerHTML = "+1.20%";
            predictionDisplay.className = "text-bullish";
        } else {
            regimeDisplay.innerHTML = "Cluster 1: Sideways Consolidation";
            regimeDisplay.className = "text-neutral";
            predictionDisplay.innerHTML = "+0.15%";
            predictionDisplay.className = "text-neutral";
        }
    }
</script>

</body>
</html>

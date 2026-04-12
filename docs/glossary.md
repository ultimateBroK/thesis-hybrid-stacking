# Glossary

> Simple definitions for every technical term used in this project. Arranged alphabetically.

---

## A

**Accuracy**
The percentage of predictions the model got correct. For 3 classes (Long/Hold/Short), random guessing gives ~33% accuracy. Anything above 45-50% is meaningful in financial prediction.

**ATR (Average True Range)**
A measure of how much the price typically moves in one period (e.g., one hour). Higher ATR means the market is more volatile. Used in this project to set Take Profit and Stop Loss levels.

**Atr_multiplier_tp / atr_multiplier_sl**
Numbers that multiply the ATR to calculate how far away the Take Profit and Stop Loss are from the current price. Default: 1.5 for both.

---

## B

**Backtest**
A simulation of trading using historical data. It shows how the strategy would have performed in the past. Past performance does not guarantee future results.

**Backtesting (CFD)**
Testing the trading strategy as a CFD (Contract for Difference) trade, which includes realistic costs like spread, slippage, and swap fees.

**Batch Size**
The number of data samples processed at once during neural network training. Larger batch sizes use more memory but can speed up training.

**Bid-Ask Spread**
The difference between the price buyers are willing to pay (bid) and the price sellers are asking (ask). This is a trading cost — you always buy slightly above the market price and sell slightly below.

**Bullish / Bearish**
- **Bullish** = expecting the price to go up (buy signal)
- **Bearish** = expecting the price to go down (sell signal)

---

## C

**Calmar Ratio**
Annual return divided by maximum drawdown. A higher Calmar ratio means better returns relative to the worst loss. Above 0.5 is reasonable.

**Candlestick (OHLCV)**
A way to show price data for a time period. Each candle has:
- **O**pen = price at the start
- **H**igh = highest price during the period
- **L**ow = lowest price during the period
- **C**lose = price at the end
- **V**olume = how much was traded

**CFD (Contract for Difference)**
A financial product where you profit (or lose) from price changes without owning the actual asset. You trade on margin with leverage.

**Class Weights**
A technique to handle imbalanced data. If "Hold" signals are much more common than "Long" signals, class weights make the model pay more attention to the rare classes.

**Confidence Threshold**
The minimum probability the model needs to make a directional prediction (Long or Short). Default: 0.6 (60%). Below this, the prediction becomes "Hold."

**Correlation Threshold**
The cutoff for removing highly correlated features. If two features are correlated above this level (default: 0.90), one is removed because it does not add new information.

**Cross-Validation**
A technique to evaluate a model by splitting data into multiple train/test combinations. This project uses **walk-forward** cross-validation that respects time ordering.

---

## D

**Data Leakage**
When information from the future accidentally leaks into the training data. This makes the model look much better than it really is. The project prevents this with **purge bars**, **embargo bars**, and careful time-based splitting.

**Drawdown**
The percentage decline from the highest point (peak) to the lowest point (trough) in your equity curve. Maximum drawdown is the worst such decline.

**Dukascopy**
A Swiss forex broker that provides free historical tick data for various currency pairs and commodities, including XAU/USD (gold).

**Dropout**
A regularization technique for neural networks. During training, a random fraction of neurons are temporarily "turned off." This prevents the network from relying too much on any single neuron. Default: 0.3 (30%).

---

## E

**Early Stopping**
A technique to stop training when the model stops improving on the validation set. Prevents overfitting. Controlled by `patience` (how many rounds to wait for improvement).

**Embargo Bars**
Extra safety bars removed after the training period to ensure no data leakage. Default: 10 bars (10 hours).

**EMA (Exponential Moving Average)**
A type of moving average that gives more weight to recent prices. Reacts faster to price changes than a simple moving average. Used periods: 34 and 89.

**Ensemble**
Combining multiple models to make predictions. This project uses a specific type called **stacking**.

---

## F

**Feature**
A piece of information the model uses to make a prediction. Examples: RSI value, EMA value, session type, lag price.

**Feature Engineering**
The process of creating new features from raw data. This project generates 20+ features from OHLCV candles.

**Feature Importance**
A ranking of which features the model relies on most. Measured using **SHAP** values.

**Fixed Fractional Position Sizing**
A risk management method where you risk a fixed percentage of your current capital on each trade. Default: 1%.

---

## G

**Gradient Boosting**
A machine learning technique that builds many decision trees sequentially, where each new tree tries to correct the mistakes of the previous trees. **LightGBM** is a gradient boosting implementation.

---

## H

**H1 (Hourly)**
A timeframe where each candle represents one hour of trading. Also written as "1H."

**Hold (class 0)**
A prediction that means "do not trade right now." The model is not confident enough to recommend buying or selling.

**Horizon Bars**
The maximum number of bars (hours) to wait for a Take Profit or Stop Loss to be hit in the Triple-Barrier method. Default: 20 bars.

**Hyperparameter**
A setting that controls how the model learns (not learned from data). Examples: learning rate, number of trees, dropout rate. Tuned manually or with **Optuna**.

---

## I

**Isotonic Regression**
A calibration method that adjusts predicted probabilities to better match actual outcomes. Used in the stacking meta-learner.

---

## K

**Kelly Criterion**
A mathematical formula that calculates the optimal fraction of capital to risk on a trade. Used for validation of backtest math.

---

## L

**L1 / L2 Regularization**
Techniques to prevent overfitting by adding a penalty for large model weights:
- **L1 (Lasso)**: Can shrink some weights to exactly zero (feature selection)
- **L2 (Ridge)**: Shrinks weights toward zero but keeps them all

**Lag Features**
Features that look at past values. For example, the close price 1, 2, 3, 5, and 10 bars ago.

**Leakage** → See **Data Leakage**

**Leverage**
Borrowing money to increase your trading position. 50:1 leverage means you control $50 for every $1 of your own capital. Amplifies both profits and losses.

**LightGBM**
A fast, efficient gradient boosting library by Microsoft. It builds decision trees that are good at finding patterns in tabular data (features).

**Long (class +1)**
A prediction to buy. You profit when the price goes up.

**LSTM (Long Short-Term Memory)**
A type of neural network designed to work with sequences of data. It can remember patterns over long sequences (120 hours in this project). Good at finding time-based patterns.

---

## M

**MACD (Moving Average Convergence Divergence)**
A momentum indicator that shows the relationship between two moving averages. When MACD is above the signal line, it suggests upward momentum. Below suggests downward momentum.

**Macro F1**
The average F1 score across all three classes (Long, Hold, Short). Better than accuracy for imbalanced data because it treats each class equally.

**Margin Call**
When your account equity falls below a certain percentage of the required margin. The broker demands you deposit more funds or they will close your positions. Default trigger: 50%.

**Market Regime**
A period of time with similar market characteristics. For example, a "low volatility regime" vs a "high volatility regime." This project splits data by market regime.

**Meta-Learner**
The model that combines predictions from the base models (LightGBM and LSTM). Default: logistic regression.

**Microstructure**
Fine-grained details about how trading happens: candlestick patterns, volume patterns, and the relationship between price movement and trading activity.

---

## N

**Normalization**
Scaling data to a standard range (usually 0 to 1 or -1 to 1). Used for the LSTM input so all features are on a similar scale.

---

## O

**OHLCV** → See **Candlestick**

**Optuna**
A library for automatic hyperparameter tuning. It tries different combinations of settings and finds the best ones. Default: 100 trials.

**OOF (Out-of-Fold) Predictions**
Predictions made on data that the model did not train on, generated during cross-validation. These are used as input to the stacking meta-learner.

**Overfitting**
When a model learns the training data too well (including noise) and performs poorly on new data. Symptoms: very high training accuracy but low test accuracy.

---

## P

**Parquet**
A columnar file format for storing data efficiently. Used throughout the project for processed data, features, and predictions.

**Patience**
In early stopping, how many rounds to wait for improvement before stopping training. Default: 10 for LSTM.

**Pip**
The smallest standard price movement for a forex pair. For XAU/USD, 1 pip = $0.01 movement in the gold price per ounce.

**Pivot Points**
Technical analysis levels calculated from the previous period's high, low, and close. Used as potential support (S1) and resistance (R1) levels.

**Platt Scaling**
A probability calibration method that fits a logistic regression to the model's raw scores. Alternative to isotonic regression.

**Polars**
A fast Python data manipulation library. Similar to Pandas but faster and uses less memory. Used throughout this project.

**Precision**
Of all the times the model predicted a class (e.g., "Long"), how many were correct? High precision = few false alarms.

**Profit Factor**
Total profits divided by total losses. Above 1.0 means the strategy is profitable overall.

**Purge Bars**
Data points removed between the training set and validation/test set to prevent information leakage. Default: 25 bars (25 hours).

---

## R

**Random Seed**
A number that controls randomness for reproducibility. Using the same seed produces the same results every time. Default: 42.

**Recall**
Of all the actual instances of a class, how many did the model find? High recall = few missed signals.

**Regression**
Predicting a continuous number. This project does **classification** (predicting categories), not regression.

**Regularization**
Techniques to prevent overfitting by penalizing complex models. Includes L1, L2, dropout, and early stopping.

**Ridge Regression**
A regression method with L2 regularization. Available as a meta-learner option.

**RSI (Relative Strength Index)**
A momentum indicator that measures the speed and magnitude of recent price changes. Ranges from 0 to 100. Above 70 suggests overbought; below 30 suggests oversold.

---

## S

**Session (Trading)**
A time period when a particular market is most active. The project recognizes three sessions: Asia (00-08 UTC), London (08-17 UTC), and NY PM (17-21 UTC).

**Session (Pipeline)**
A single pipeline run with all its outputs stored in a timestamped folder under `results/`.

**SHAP (SHapley Additive exPlanations)**
A method to explain machine learning predictions. It shows how much each feature contributed to each prediction. The SHAP summary plot ranks features by overall importance.

**Sharpe Ratio**
A measure of risk-adjusted return. Calculated as: (average return - risk-free rate) / standard deviation of returns. Higher is better. Above 1.0 is generally considered good.

**Short (class -1)**
A prediction to sell. You profit when the price goes down.

**Slippage**
The difference between the expected trade price and the actual execution price. Always a cost to the trader. Default: 1 pip.

**Sortino Ratio**
Similar to the Sharpe Ratio but only considers downside volatility (losses). Should be higher than the Sharpe ratio because upside volatility is not penalized.

**Spread**
The difference between the bid and ask price. A trading cost. Default: 2 pips for XAU/USD.

**Stacking**
An ensemble technique where multiple base models make predictions, and a meta-learner combines them. Better than simple voting because the meta-learner learns which model to trust when.

**Stop Loss (SL)**
A price level where a losing trade is automatically closed to limit losses. In the Triple-Barrier method: Close - ATR_multiplier x ATR.

**Stop-Out**
When the broker forcibly closes all positions because the account equity is too low. Default trigger: 20% of margin.

---

## T

**TA-Lib**
A popular C library for calculating technical indicators. This project uses it for EMA, RSI, MACD, and ATR calculations.

**Take Profit (TP)**
A price level where a winning trade is automatically closed to lock in profits. In the Triple-Barrier method: Close + ATR_multiplier x ATR.

**Test Set**
The final dataset used to evaluate the model. The model never sees this data during training. In this project: 2024-01-01 to 2026-03-31.

**Timeframe**
The duration of each candle. This project uses 1H (1 hour). Other common timeframes: 15T (15 minutes), 4H (4 hours), D1 (daily).

**TOML**
A configuration file format (Tom's Obvious Minimal Language). Simple and readable. The project stores all settings in `config.toml`.

**Trailing Stop**
A stop-loss level that follows the price as it moves in your favor. If the price reverses, the stop stays at its last level, protecting profits.

**Train Set**
The dataset used to teach the model. In this project: 2018-01-01 to 2022-12-31.

**Triple-Barrier Method**
A labeling technique that places three "barriers" around the current price: Take Profit (above), Stop Loss (below), and a time limit. The label depends on which barrier is hit first.

---

## V

**Validation Set**
A dataset used during training to monitor performance and prevent overfitting. Not used for training itself. In this project: 2023-01-01 to 2023-12-31.

---

## W

**Walk-Forward Cross-Validation**
A time-series validation method where the training window moves forward in time. Each fold uses older data for training and newer data for validation. Respects time ordering.

**Win Rate**
The percentage of trades that were profitable. Not the only important metric — a strategy with 40% win rate can be profitable if winners are much larger than losers.

---

## X

**XAU/USD**
The ticker symbol for trading gold (XAU) priced in US dollars (USD). Also known as "spot gold."

# NBA Predictor

> Machine learning system for NBA game and player performance prediction

## 🎯 Performance
- **72.4%** winner accuracy (beats home bias by 16.3pp)
- **9.78 pts** Mean Absolute Error
- **<2%** train/test gap (no overfitting)

## 🏗️ Architecture
[Add a simple diagram: Data Ingestion → Feature Engineering → Model Training → Predictions]

## 📊 Key Features
- Automated daily pipeline (fetch games, compute rolling stats, generate predictions)
- Injury-aware predictions (ESPN API integration)
- Playoff context modeling (series desperation, elimination games)
- Player performance forecasting (points MAE: 4.93, rebounds: 2.01, assists: 1.53)

## 🛠️ Tech Stack
Python | PostgreSQL | SQLAlchemy | scikit-learn | NBA API | Streamlit

## 📈 Sample Prediction
[Screenshot of Streamlit dashboard showing a game prediction]

## 🚀 What I Learned
- Chronological splitting prevents data leakage in time-series prediction
- Playoff efficiency (points-per-usage) beats traditional momentum for slump detection
- Ensemble models with dynamic weights outperform single models by 4.2%
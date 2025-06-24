# 🎯 Valorant Rank Predictor

This project uses a machine learning model to **predict the rank of a Valorant player** based on their in-game statistics such as kills, deaths, assists, headshots, damage, etc.

A mini web demo using **Streamlit** is included for easy experimentation.

---

## 🚀 Project Goals

- Build a model to predict a player's rank from match statistics.
- Explore the correlation between individual performance and rank.
- Practice a full ML pipeline: preprocessing → training → evaluation → web deployment.

---

## 🧠 Model Used

- `RandomForestRegressor` (regression) with post-processing `.round()` to convert predictions into discrete rank classes.
- Feature engineering includes:
  - KDA ratio
  - Headshot rate
  - Damage per match
  - Survival rate, etc.

---

## 📊 Result Insights

> The confusion matrix shows that the model performs well for lower (**iron**, **bronze**) and higher (**immortal**) ranks but struggles with middle tiers such as **silver → platinum**.  
> This reflects real gameplay, where mid ranks vary due to **inconsistency in performance or team composition**.  
> Despite the challenge, the model achieves **average recall around 50%** with a solid **R² ≈ 0.76**, indicating decent overall predictive power.

---

## 📁 Folder Structure

```
valorant_rank_predictor/
├── app.py                 # Streamlit web interface
├── model/
│   └── rf_model.pkl       # Trained model file
├── data/
│   └── valorant_dataset.csv  # (optional) source data
├── requirements.txt       # Required libraries
└── README.md              # Project description
```

---

## ▶️ How to Run

### 1. Install dependencies
```bash
pip install -r requirements.txt
```

### 2. Launch the web app
```bash
streamlit run app.py
```

---

## 🛠️ Techniques Applied

- Data cleaning: numeric conversion, normalization
- Feature engineering: statistical ratios and per-match metrics
- GridSearchCV: hyperparameter tuning for RandomForest
- Confusion matrix: evaluation by rank class
- Streamlit: rapid web demo deployment

---

## 📌 Notes

- The dataset is not severely imbalanced but has slightly more samples in higher ranks.
- Performance can be further improved by using more data or incorporating map/agent/team-based features.

---

## 💡 Future Ideas

- Track rank progression over time
- Predict win rate or MVP chance
- Recommend optimal agent based on current stats

---

## 👨‍💻 Author

- Name: [Your Name]
- Contact: [Email / GitHub / LinkedIn]
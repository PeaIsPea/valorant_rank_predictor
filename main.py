import pandas as pd
import numpy as np
from sklearn.compose import ColumnTransformer
from sklearn.pipeline import Pipeline
from sklearn.model_selection import train_test_split, GridSearchCV
from sklearn.ensemble import RandomForestRegressor
from sklearn.preprocessing import StandardScaler
from sklearn.metrics import classification_report
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.metrics import confusion_matrix
import joblib

# ===== 1. Load và xử lý dữ liệu gốc =====
data = pd.read_csv("data/valorant_dataset.csv")
target = "rank"
x = data.drop(target, axis=1)
y_raw = data[target]

# Chuyển rank thành số thứ tự (ordinal)
rank_order = ['iron', 'bronze', 'silver', 'gold', 'platinum', 'diamond', 'ascendant', 'immortal']
rank_map = {r: i for i, r in enumerate(rank_order)}
y = y_raw.map(rank_map)

# Fix các cột số bị lưu thành chuỗi
cols_to_fix = ['assists', 'damage_received']
for col in cols_to_fix:
    x[col] = x[col].str.replace(',', '', regex=False)
    x[col] = pd.to_numeric(x[col])

# ===== 2. Feature Engineering =====
x['kda_ratio'] = (x['kills'] + x['assists']) / (x['deaths'] + 1e-5)
x['hs_rate'] = x['headshots'] / (x['kills'] + 1e-5)
x['dmg_per_match'] = x['damage'] / (x['matches'] + 1e-5)
x['dmg_taken_per_death'] = x['damage_received'] / (x['deaths'] + 1e-5)
x['survival_rate'] = 1 - x['deaths'] / (x['matches'] + 1e-5)
x['avg_assists'] = x['assists'] / (x['matches'] + 1e-5)
x['avg_traded'] = x['traded'] / (x['kills'] + 1e-5)

# ===== 3. Chia tập train/test =====
x_train, x_test, y_train, y_test = train_test_split(
    x, y, test_size=0.2, random_state=42, stratify=y
)

# ===== 4. Pipeline tiền xử lý =====
num_transformer = Pipeline([
    ("scaler", StandardScaler())
])

preprocessor = ColumnTransformer([
    ("num", num_transformer, x.columns.tolist())
])

# ===== 5. Pipeline mô hình RandomForestRegressor =====
regressor = Pipeline([
    ("preprocessor", preprocessor),
    ("model", RandomForestRegressor(random_state=42))
])

# ===== 6. GridSearch để tìm tham số tốt nhất =====
param_grid = {
    "model__n_estimators": [100, 200, 300],
    "model__max_depth": [6, 10, 20]
}

grid_search = GridSearchCV(
    regressor,
    param_grid=param_grid,
    cv=4,
    scoring='r2',
    verbose=2,
    n_jobs=-1
)

grid_search.fit(x_train, y_train)

# ===== 7. Dự đoán & làm tròn thành rank =====
y_pred_float = grid_search.predict(x_test)
y_pred_class = np.round(y_pred_float).astype(int).clip(0, len(rank_order) - 1)

# ===== 8. Đánh giá kết quả phân loại =====
print("Best R2 score:", grid_search.best_score_)
print("Best params:", grid_search.best_params_)
print(classification_report(y_test, y_pred_class, target_names=rank_order))


# Lưu mô hình sau khi train
joblib.dump(grid_search.best_estimator_, "model/rf_model.pkl")

# Tạo confusion matrix
cm = confusion_matrix(y_test, y_pred_class)
# Vẽ bằng seaborn
plt.figure(figsize=(10, 8))
sns.heatmap(cm, annot=True, fmt="d", cmap="Blues", xticklabels=rank_order, yticklabels=rank_order)

# Nhãn trục
plt.xlabel("Predicted Rank")
plt.ylabel("True Rank")
plt.title("Confusion Matrix for Valorant Rank Prediction")
plt.tight_layout()
plt.show()

import pandas as pd
import numpy as np
from sklearn.compose import ColumnTransformer
from sklearn.pipeline import Pipeline
from sklearn.model_selection import train_test_split, GridSearchCV
from sklearn.ensemble import RandomForestClassifier
from sklearn.preprocessing import StandardScaler
from sklearn.metrics import classification_report, confusion_matrix
import matplotlib.pyplot as plt
import seaborn as sns
import joblib
from imblearn.over_sampling import SMOTE

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


# ===== 2. Chia tập train/test =====
x_train, x_test, y_train, y_test = train_test_split(
    x, y, test_size=0.2, random_state=42, stratify=y
)


# ===== 3. Dùng SMOTE để over-sample tập train =====
smote = SMOTE(random_state=42, sampling_strategy={
    0: 300,  # iron
    1: 300,  # bronze
    2: 300,  # silver
    3: 300,  # gold
    4: 300,  # platinum
    5: 400,  # diamond

})
print(y_train.value_counts())
x_train, y_train= smote.fit_resample(x_train, y_train)
print("-------------------------")
print(y_train.value_counts())





# ===== 4. Pipeline tiền xử lý =====
num_transformer = Pipeline([
    ("scaler", StandardScaler())
])

preprocessor = ColumnTransformer([
    ("num", num_transformer, x.columns.tolist())
])

# ===== 5. Pipeline mô hình RandomForestClassifier =====
classifier = Pipeline([
    ("preprocessor", preprocessor),
    ("model", RandomForestClassifier(random_state=42))
])

# ===== 6. GridSearch để tìm tham số tốt nhất =====
param_grid = {

    "model__n_estimators": [100, 200, 300],
    "model__max_depth": [6, 10, 20],
    "model__class_weight": [None, "balanced"]
}

grid_search = GridSearchCV(
    classifier,
    param_grid=param_grid,
    cv=4,
    scoring='f1_macro',
    verbose=2,
    n_jobs=-1
)

grid_search.fit(x_train, y_train)

# ===== 7. Dự đoán =====
y_pred = grid_search.predict(x_test)

# ===== 8. Đánh giá kết quả phân loại =====
print("Best F1 score:", grid_search.best_score_)
print("Best params:", grid_search.best_params_)
print(classification_report(y_test, y_pred, target_names=rank_order))

# ===== 9. Lưu mô hình đã huấn luyện =====
joblib.dump(grid_search.best_estimator_, "model/rf_classifier.pkl")

# ===== 10. Vẽ confusion matrix =====
cm = confusion_matrix(y_test, y_pred)
plt.figure(figsize=(10, 8))
sns.heatmap(cm, annot=True, fmt="d", cmap="Blues", xticklabels=rank_order, yticklabels=rank_order)
plt.xlabel("Predicted Rank")
plt.ylabel("True Rank")
plt.title("Confusion Matrix with SMOTE + RandomForestClassifier")
plt.tight_layout()
plt.show()



# accuracy                           0.56       539
# macro avg       0.59      0.58      0.58       539
# weighted avg       0.58      0.56      0.57       539
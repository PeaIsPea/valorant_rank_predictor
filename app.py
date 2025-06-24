# app.py
import streamlit as st
import pandas as pd
import numpy as np
import joblib

# Load model
model = joblib.load("model/rf_model.pkl")

# Rank map
rank_order = ['iron', 'bronze', 'silver', 'gold', 'platinum', 'diamond', 'ascendant', 'immortal']

# Streamlit UI
st.title("Valorant Rank Predictor")
st.write("Dự đoán rank của bạn dựa trên thống kê trận đấu")

# Nhập thông số người dùng
kills = st.number_input("Kills", min_value=0, value=10)
deaths = st.number_input("Deaths", min_value=0, value=5)
assists = st.number_input("Assists", min_value=0, value=3)
headshots = st.number_input("Headshots", min_value=0, value=5)
damage = st.number_input("Total Damage", min_value=0, value=15000)
damage_received = st.number_input("Damage Received", min_value=0, value=12000)
matches = st.number_input("Matches Played", min_value=1, value=10)
traded = st.number_input("Traded Kills", min_value=0, value=2)

# Khi nhấn nút dự đoán
if st.button("Dự đoán rank"):
    # Tính toán lại các feature như lúc train
    x_input = pd.DataFrame([{
        "kills": kills,
        "deaths": deaths,
        "assists": assists,
        "headshots": headshots,
        "damage": damage,
        "damage_received": damage_received,
        "matches": matches,
        "traded": traded,
        "kda_ratio": (kills + assists) / (deaths + 1e-5),
        "hs_rate": headshots / (kills + 1e-5),
        "dmg_per_match": damage / (matches + 1e-5),
        "dmg_taken_per_death": damage_received / (deaths + 1e-5),
        "survival_rate": 1 - deaths / (matches + 1e-5),
        "avg_assists": assists / (matches + 1e-5),
        "avg_traded": traded / (kills + 1e-5)
    }])

    # Dự đoán và hiển thị kết quả
    pred = model.predict(x_input)
    pred_class = int(np.round(pred).clip(0, len(rank_order) - 1))
    st.success(f"✅ Rank dự đoán: **{rank_order[pred_class].capitalize()}**")

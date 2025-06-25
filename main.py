from flask import Flask, request, jsonify
from catboost import CatBoostRegressor
import pandas as pd
import numpy as np
import json
import datetime

# 🚀 Initialize app
app = Flask(__name__)

# 📦 Load model
model = CatBoostRegressor()
model.load_model("catboost_model.cbm")

# 📄 Load metadata
with open("columns.json", "r") as f:
    meta = json.load(f)

column_order = meta["columns"]
cat_features = meta["cat_features"]

# 🔍 Preprocessing function (ใช้ Year ตรง ๆ แล้ว)
def preprocess_input(data: dict):
    df = pd.DataFrame([data])

    # ✅ ไม่แปลง Year แล้ว ใช้ตรง ๆ เลย
    # ✅ ไม่ต้องลบ Year_Built แล้ว เพราะไม่มีใน input แล้ว

    # จัดเรียงคอลัมน์ให้ตรง
    for col in column_order:
        if col not in df.columns:
            df[col] = np.nan  # ใส่ค่าว่างให้ column ที่ไม่มีใน input

    df = df[column_order]  # เรียงให้ตรงกับตอน train
    return df

# 🎯 Route for prediction
@app.route("/predict", methods=["POST"])
def predict():
    try:
        data = request.get_json()
        X_input = preprocess_input(data)
        pred = model.predict(X_input)[0]
        return jsonify({"prediction": round(pred, 2)})
    except Exception as e:
        return jsonify({"error": str(e)})

# 🏁 Run the app (for local testing only)
if __name__ == "__main__":
    import os
    port = int(os.environ.get("PORT", 5000))
    app.run(host="0.0.0.0", port=port)

"""Predict churn for new customers.
Usage:  python predict_new.py new_customers.csv predictions.csv
"""
import sys
import joblib
import numpy as np
import pandas as pd

SERVICE_COLS = ["PhoneService", "MultipleLines", "OnlineSecurity", "OnlineBackup",
                "DeviceProtection", "TechSupport", "StreamingTV", "StreamingMovies"]


def add_features(d):
    """Same 5 features as in training."""
    d = d.copy()
    d["tenure_group"] = pd.cut(d["tenure"], bins=[-1, 12, 24, 48, 72],
                               labels=["0-1 yr", "1-2 yr", "2-4 yr", "4-6 yr"]).astype(str)
    d["avg_monthly_spend"] = np.where(d["tenure"] > 0,
                                      d["TotalCharges"] / d["tenure"].clip(lower=1),
                                      d["MonthlyCharges"])
    d["charge_increase"] = d["MonthlyCharges"] - d["avg_monthly_spend"]
    d["num_services"] = (d[SERVICE_COLS] == "Yes").sum(axis=1)
    d["auto_payment"] = d["PaymentMethod"].str.contains("automatic").astype(int)
    return d


def prepare(raw):
    """Clean raw data exactly as in training."""
    d = raw.copy()
    d["TotalCharges"] = pd.to_numeric(d["TotalCharges"], errors="coerce").fillna(0)
    ids = d["customerID"] if "customerID" in d.columns else pd.Series(range(len(d)))
    d = d.drop(columns=["customerID", "Churn"], errors="ignore")
    return ids, add_features(d)


def main():
    in_path = sys.argv[1] if len(sys.argv) > 1 else "new_customers.csv"
    out_path = sys.argv[2] if len(sys.argv) > 2 else "predictions.csv"

    saved = joblib.load("churn_model.joblib")
    model, threshold = saved["model"], saved["threshold"]

    ids, X = prepare(pd.read_csv(in_path))
    proba = model.predict_proba(X)[:, 1]

    out = pd.DataFrame({"customerID": ids.values,
                        "churn_probability": proba.round(3),
                        "predicted_churn": (proba >= threshold).astype(int)})
    out["action"] = np.where(out["predicted_churn"] == 1,
                             "Call - retention offer", "No action")
    out = out.sort_values("churn_probability", ascending=False)
    out.to_csv(out_path, index=False)

    print(f"Scored {len(out)} customers with threshold {threshold:.3f}")
    print(f"Flagged for a call: {out['predicted_churn'].sum()}")
    print(f"Saved {out_path}")


if __name__ == "__main__":
    main()

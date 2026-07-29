import pandas as pd
import numpy as np

RAW = "/sessions/eloquent-friendly-davinci/mnt/uploads/Airline_Delay_Cause.csv"
df = pd.read_csv(RAW)
print("Raw shape:", df.shape)
print(df.isna().sum())

num_cols = ["arr_flights","arr_del15","carrier_ct","weather_ct","nas_ct","security_ct",
            "late_aircraft_ct","arr_cancelled","arr_diverted","arr_delay","carrier_delay",
            "weather_delay","nas_delay","security_delay","late_aircraft_delay"]

for c in num_cols:
    df[c] = pd.to_numeric(df[c], errors="coerce").fillna(0)
    df[c] = df[c].clip(lower=0)

df["year"] = df["year"].astype(int)
df["month"] = df["month"].astype(int)

# cap extreme delay-minute fields at 99th percentile (matches documented approach)
for c in ["arr_delay","carrier_delay","weather_delay","nas_delay","security_delay","late_aircraft_delay"]:
    cap = df[c].quantile(0.99)
    df[c] = df[c].clip(upper=cap)

# feature engineering
df["effective_arrivals"] = df["arr_flights"] - df["arr_cancelled"] - df["arr_diverted"]
df["effective_arrivals"] = df["effective_arrivals"].clip(lower=0)
df["delay_rate"] = np.where(df["arr_flights"]>0, df["arr_del15"]/df["arr_flights"], 0)
df["cancellation_rate"] = np.where(df["arr_flights"]>0, df["arr_cancelled"]/df["arr_flights"], 0)
df["total_delay_minutes"] = df["arr_delay"]
df["delay_hours"] = df["total_delay_minutes"]/60.0
df["avg_delay_per_delayed_flight"] = np.where(df["arr_del15"]>0, df["total_delay_minutes"]/df["arr_del15"], 0)
df["month_ts"] = pd.to_datetime(df["year"].astype(str)+"-"+df["month"].astype(str)+"-01")

df.to_csv("/tmp/airline/data/cleaned/airline_delay_cleaned.csv", index=False)
print("Cleaned shape:", df.shape)
print(df.describe().T[["mean","std","min","max"]])

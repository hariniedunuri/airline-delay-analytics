import pandas as pd, numpy as np
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import seaborn as sns

sns.set_style("whitegrid")
plt.rcParams["figure.dpi"] = 120

df = pd.read_csv("/tmp/airline/data/cleaned/airline_delay_cleaned.csv", parse_dates=["month_ts"])
OUT = "/tmp/airline/visualizations"

# 1. Distribution of arrival delay (row-level minutes, exclude zeros for readability)
plt.figure(figsize=(8,5))
sns.histplot(df[df["arr_delay"]>0]["arr_delay"], bins=60, color="#3b6fa0")
plt.title("Distribution of Monthly Arrival Delay Minutes (per Airline-Airport)")
plt.xlabel("Total Arrival Delay Minutes"); plt.ylabel("Count")
plt.tight_layout(); plt.savefig(f"{OUT}/01_delay_distribution.png"); plt.close()

# 2. Monthly delay rate trend (system-wide)
monthly = df.groupby("month_ts").apply(lambda g: g["arr_del15"].sum()/g["arr_flights"].sum()).reset_index(name="delay_rate")
plt.figure(figsize=(10,5))
plt.plot(monthly["month_ts"], monthly["delay_rate"], color="#c0392b")
plt.title("System-Wide Monthly Delay Rate (2003-2025)")
plt.xlabel("Month"); plt.ylabel("Share of Flights Delayed 15+ min")
plt.tight_layout(); plt.savefig(f"{OUT}/02_monthly_delay_rate_trend.png"); plt.close()

# 3. Average delay hours by year
yearly = df.groupby("year")["delay_hours"].sum().reset_index()
plt.figure(figsize=(9,5))
sns.barplot(data=yearly, x="year", y="delay_hours", color="#2e6f95")
plt.title("Total System Delay Hours by Year")
plt.xticks(rotation=45); plt.ylabel("Total Delay Hours"); plt.xlabel("Year")
plt.tight_layout(); plt.savefig(f"{OUT}/03_delay_hours_by_year.png"); plt.close()

# 4. Top 10 airlines by avg delay per delayed flight (min 5000 delayed flights to avoid tiny-sample noise)
carrier_agg = df.groupby("carrier_name").agg(delayed=("arr_del15","sum"), minutes=("total_delay_minutes","sum")).reset_index()
carrier_agg = carrier_agg[carrier_agg["delayed"]>5000]
carrier_agg["avg_delay_per_flight"] = carrier_agg["minutes"]/carrier_agg["delayed"]
top10_air = carrier_agg.sort_values("avg_delay_per_flight", ascending=False).head(10)
plt.figure(figsize=(9,6))
sns.barplot(data=top10_air, y="carrier_name", x="avg_delay_per_flight", color="#8e44ad")
plt.title("Top 10 Airlines by Average Delay per Delayed Flight (min.)")
plt.xlabel("Avg Minutes per Delayed Flight"); plt.ylabel("")
plt.tight_layout(); plt.savefig(f"{OUT}/04_top10_airlines_avg_delay.png"); plt.close()

# 5. Top 10 airports by average arrival delay rate (min 5000 total flights)
airport_agg = df.groupby("airport_name").agg(flights=("arr_flights","sum"), delayed=("arr_del15","sum")).reset_index()
airport_agg = airport_agg[airport_agg["flights"]>5000]
airport_agg["delay_rate"] = airport_agg["delayed"]/airport_agg["flights"]
top10_air_port = airport_agg.sort_values("delay_rate", ascending=False).head(10)
plt.figure(figsize=(9,6))
sns.barplot(data=top10_air_port, y="airport_name", x="delay_rate", color="#16a085")
plt.title("Top 10 Airports by Arrival Delay Rate")
plt.xlabel("Share of Flights Delayed 15+ min"); plt.ylabel("")
plt.tight_layout(); plt.savefig(f"{OUT}/05_top10_airports_delay_rate.png"); plt.close()

# 6. Delay causes across months (share of total delay minutes)
cause_cols = ["carrier_delay","weather_delay","nas_delay","security_delay","late_aircraft_delay"]
by_month = df.groupby("month")[cause_cols].sum()
by_month_share = by_month.div(by_month.sum(axis=1), axis=0)
plt.figure(figsize=(10,6))
by_month_share.plot(kind="bar", stacked=True, ax=plt.gca(), colormap="tab10")
plt.title("Share of Delay Minutes by Cause, Across Months")
plt.xlabel("Month"); plt.ylabel("Share of Total Delay Minutes")
plt.legend(title="Cause", bbox_to_anchor=(1.02,1), loc="upper left")
plt.tight_layout(); plt.savefig(f"{OUT}/06_delay_causes_by_month.png"); plt.close()

print("EDA charts written:")
import os
for f in sorted(os.listdir(OUT)):
    print(" -", f)

print("\nKey EDA numbers:")
print("Overall delay rate:", round(df["arr_del15"].sum()/df["arr_flights"].sum(),4))
print("Cause share of total delay minutes:")
print((df[cause_cols].sum()/df[cause_cols].sum().sum()).round(3))

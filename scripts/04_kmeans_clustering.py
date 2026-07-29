import pandas as pd, numpy as np
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
from sklearn.preprocessing import StandardScaler
from sklearn.cluster import KMeans

df = pd.read_csv("/tmp/airline/data/cleaned/airline_delay_cleaned.csv")

agg = df.groupby("carrier_name").agg(
    flights=("arr_flights","sum"),
    delayed=("arr_del15","sum"),
    delay_hours=("delay_hours","sum"),
    cancelled=("arr_cancelled","sum"),
    carrier_delay=("carrier_delay","sum"),
    weather_delay=("weather_delay","sum"),
    nas_delay=("nas_delay","sum"),
    late_aircraft_delay=("late_aircraft_delay","sum"),
).reset_index()

# only keep carriers with a meaningful volume (drop tiny/regional codeshare noise)
agg = agg[agg["flights"] > 20000].copy()
agg["delay_rate"] = agg["delayed"]/agg["flights"]
agg["cancellation_rate"] = agg["cancelled"]/agg["flights"]
total_cause = agg[["carrier_delay","weather_delay","nas_delay","late_aircraft_delay"]].sum(axis=1)
agg["late_aircraft_share"] = agg["late_aircraft_delay"]/total_cause
agg["delay_hours_per_1k_flights"] = agg["delay_hours"]/(agg["flights"]/1000)

features = ["delay_rate","cancellation_rate","late_aircraft_share","delay_hours_per_1k_flights"]
X = agg[features].fillna(0)
Xs = StandardScaler().fit_transform(X)

inertias = []
for k in range(1,8):
    inertias.append(KMeans(n_clusters=k, n_init=10, random_state=42).fit(Xs).inertia_)
plt.figure(figsize=(6,4))
plt.plot(range(1,8), inertias, marker="o")
plt.title("Elbow Method for K Selection"); plt.xlabel("k"); plt.ylabel("Inertia")
plt.tight_layout(); plt.savefig("/tmp/airline/visualizations/08_kmeans_elbow.png"); plt.close()

km = KMeans(n_clusters=3, n_init=10, random_state=42).fit(Xs)
agg["cluster"] = km.labels_

summary = agg.groupby("cluster")[features + ["flights"]].mean().round(3)
print(summary)
print(agg[["carrier_name","cluster","delay_rate","cancellation_rate"]].sort_values("cluster"))

plt.figure(figsize=(8,6))
palette = {0:"#2ecc71",1:"#e74c3c",2:"#3498db"}
for c in sorted(agg["cluster"].unique()):
    sub = agg[agg["cluster"]==c]
    plt.scatter(sub["delay_rate"], sub["delay_hours_per_1k_flights"], label=f"Cluster {c}",
                color=palette.get(c,"#333"), s=80)
    for _, r in sub.iterrows():
        plt.annotate(r["carrier_name"].split()[0], (r["delay_rate"], r["delay_hours_per_1k_flights"]), fontsize=7)
plt.xlabel("Delay Rate"); plt.ylabel("Delay Hours per 1,000 Flights")
plt.title("K-Means Airline Clusters (k=3)")
plt.legend()
plt.tight_layout()
plt.savefig("/tmp/airline/visualizations/09_kmeans_clusters.png")
plt.close()

agg.to_csv("/tmp/airline/data/cleaned/airline_clusters.csv", index=False)
print("Saved cluster chart + elbow chart + airline_clusters.csv")

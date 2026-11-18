from pathlib import Path
import pandas as pd
import matplotlib.pyplot as plt

ROOT = Path(__file__).resolve().parents[1]
DATA = ROOT / "data" / "market_prices_illustrative.csv"
OUT = ROOT / "visualizations"
OUT.mkdir(exist_ok=True)

df = pd.read_csv(DATA)
print("Original shape:", df.shape)
print("Missing values:\n", df.isna().sum())
print("Duplicate rows:", df.duplicated().sum())

df["date"] = pd.to_datetime(df["date"], errors="coerce")
df["price_ngn"] = pd.to_numeric(df["price_ngn"], errors="coerce")
df = df.drop_duplicates().dropna(subset=["date", "state", "market", "commodity", "price_ngn"])
df = df[df["price_ngn"] > 0].copy()
df["month"] = df["date"].dt.to_period("M").astype(str)

commodity_summary = df.groupby("commodity")["price_ngn"].agg(["mean", "median", "min", "max", "count"]).sort_values("mean", ascending=True)
commodity_summary.to_csv(OUT / "commodity_summary.csv")
print("\nCommodity summary:\n", commodity_summary.round(2))

market_summary = df.groupby(["state", "market", "commodity"])["price_ngn"].mean().reset_index()
market_summary.to_csv(OUT / "market_commodity_summary.csv", index=False)

fig, ax = plt.subplots(figsize=(10, 6))
ax.barh(commodity_summary.index, commodity_summary["mean"])
ax.set_title("Average Illustrative Price by Commodity")
ax.set_xlabel("Average price (NGN)")
fig.tight_layout()
fig.savefig(OUT / "average_price_by_commodity.png", dpi=160)
plt.close(fig)

monthly = df.groupby(["month", "commodity"])["price_ngn"].mean().unstack()
fig, ax = plt.subplots(figsize=(12, 6))
for col in monthly.columns:
    ax.plot(monthly.index, monthly[col], label=col)
ax.set_title("Illustrative Monthly Commodity Price Trends")
ax.set_xlabel("Month")
ax.set_ylabel("Average price (NGN)")
ax.tick_params(axis="x", rotation=45)
ax.legend()
fig.tight_layout()
fig.savefig(OUT / "monthly_price_trends.png", dpi=160)
plt.close(fig)

state_avg = df.groupby("state")["price_ngn"].mean().sort_values()
fig, ax = plt.subplots(figsize=(9, 5))
ax.barh(state_avg.index, state_avg.values)
ax.set_title("Average Illustrative Prices by State")
ax.set_xlabel("Average price (NGN)")
fig.tight_layout()
fig.savefig(OUT / "average_price_by_state.png", dpi=160)
plt.close(fig)

pivot = df.pivot_table(index="state", columns="commodity", values="price_ngn", aggfunc="mean")
fig, ax = plt.subplots(figsize=(11, 6))
im = ax.imshow(pivot.values, aspect="auto")
ax.set_xticks(range(len(pivot.columns)), pivot.columns, rotation=35, ha="right")
ax.set_yticks(range(len(pivot.index)), pivot.index)
ax.set_title("Average Illustrative Commodity Prices by State")
fig.colorbar(im, ax=ax, label="Average price (NGN)")
fig.tight_layout()
fig.savefig(OUT / "state_commodity_heatmap.png", dpi=160)
plt.close(fig)

print("\nCharts saved to:", OUT)
print("NOTE: Synthetic demonstration data only; not verified market observations.")

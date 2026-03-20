import pandas as pd
import matplotlib.pyplot as plt
import matplotlib.ticker as mticker
from collections import Counter
import re

df = pd.read_csv("src/merged_companies.csv")

# ── Shared style ──────────────────────────────────────────────────────────────
plt.rcParams.update({
    "font.family": "sans-serif",
    "axes.spines.top": False,
    "axes.spines.right": False,
    "axes.titlesize": 13,
    "axes.labelsize": 11,
})
BLUE = "#4C72B0"
COLORS = ["#4C72B0", "#DD8452", "#55A868", "#C44E52", "#8172B2", "#937860"]


# ─────────────────────────────────────────────────────────────────────────────
# PLOT 1: Founding year distribution (histogram)
# ─────────────────────────────────────────────────────────────────────────────
fig, ax = plt.subplots(figsize=(9, 4.5))

years = df["year_founded"].dropna()
years = years[(years >= 2000) & (years <= 2023)].astype(int)

ax.hist(years, bins=range(2000, 2025), color=BLUE, edgecolor="white", linewidth=0.6, rwidth=0.85)
ax.set_title("Distribution of Startups by Founding Year")
ax.set_xlabel("Year Founded")
ax.set_ylabel("Number of Startups")
ax.xaxis.set_major_locator(mticker.MultipleLocator(2))
ax.tick_params(axis="x", rotation=45)

# annotate peak
peak_year = years.value_counts().idxmax()
peak_count = years.value_counts().max()
ax.annotate(f"Peak: {peak_year}\n(n={peak_count})",
            xy=(peak_year, peak_count),
            xytext=(peak_year - 4, peak_count - 30),
            arrowprops=dict(arrowstyle="->", color="grey"),
            fontsize=9, color="grey")

fig.tight_layout()
fig.savefig("src/plot_founding_year.png", dpi=150)
plt.close(fig)
print("Saved plot_founding_year.png")


# ─────────────────────────────────────────────────────────────────────────────
# PLOT 2: Top 15 startup tags / categories (horizontal bar chart)
# ─────────────────────────────────────────────────────────────────────────────
all_tags = []
for entry in df["tags"].dropna():
    all_tags.extend([t.strip() for t in re.split(r"[|,]", entry)])

tag_counts = Counter(all_tags)
top_tags = pd.Series(dict(tag_counts.most_common(15))).sort_values()

fig, ax = plt.subplots(figsize=(9, 6))
bars = ax.barh(top_tags.index, top_tags.values, color=BLUE, edgecolor="white")

for bar in bars:
    ax.text(bar.get_width() + 8, bar.get_y() + bar.get_height() / 2,
            str(int(bar.get_width())), va="center", fontsize=9, color="#333")

ax.set_title("Top 15 Most Common Startup Categories / Tags")
ax.set_xlabel("Number of Startups")
ax.set_xlim(0, top_tags.max() * 1.15)

fig.tight_layout()
fig.savefig("src/plot_top_tags.png", dpi=150)
plt.close(fig)
print("Saved plot_top_tags.png")


# ─────────────────────────────────────────────────────────────────────────────
# PLOT 3: Company status breakdown (pie chart)
# ─────────────────────────────────────────────────────────────────────────────
status_map = {
    "Active": "Active",
    "Operating": "Active",
    "Public": "Public",
    "Acquired": "Acquired",
    "Exited": "Acquired",
    "Inactive": "Inactive",
    "Dead": "Inactive",
}
status = df["status"].map(status_map).fillna("Unknown")
status_counts = status.value_counts()

fig, ax = plt.subplots(figsize=(6, 5))
wedges, texts, autotexts = ax.pie(
    status_counts,
    labels=status_counts.index,
    autopct="%1.1f%%",
    colors=COLORS[:len(status_counts)],
    startangle=140,
    pctdistance=0.78,
    wedgeprops=dict(edgecolor="white", linewidth=1.5),
)
for t in autotexts:
    t.set_fontsize(9)
ax.set_title("Startup Status Distribution")

fig.tight_layout()
fig.savefig("src/plot_status_pie.png", dpi=150)
plt.close(fig)
print("Saved plot_status_pie.png")


# ─────────────────────────────────────────────────────────────────────────────
# PLOT 4: Missing value heatmap for key columns
# ─────────────────────────────────────────────────────────────────────────────
key_cols = [
    "canonical_name", "short_description", "long_description",
    "tags", "year_founded", "country", "status",
    "team_size", "num_founders", "founders_names",
]
missing_pct = df[key_cols].isnull().mean() * 100
missing_pct = missing_pct.sort_values(ascending=True)

fig, ax = plt.subplots(figsize=(9, 4.5))
bars = ax.barh(missing_pct.index, missing_pct.values,
               color=["#55A868" if v < 10 else "#DD8452" if v < 40 else "#C44E52"
                      for v in missing_pct.values],
               edgecolor="white")

for bar in bars:
    val = bar.get_width()
    ax.text(val + 0.5, bar.get_y() + bar.get_height() / 2,
            f"{val:.1f}%", va="center", fontsize=9)

ax.set_title("Missing Data Rate for Key Columns")
ax.set_xlabel("% Missing")
ax.set_xlim(0, 110)
ax.axvline(x=0, color="grey", linewidth=0.5)

fig.tight_layout()
fig.savefig("src/plot_missing_data.png", dpi=150)
plt.close(fig)
print("Saved plot_missing_data.png")

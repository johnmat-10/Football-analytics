# -*- coding: utf-8 -*-
"""
Goalkeeper Performance Analysis & Profiling
===========================================
1. Performance Scatter: Shot Stopping vs Passing (with sample size warnings)
2. Profile Heatmap: Top 20 Goalkeepers ranked by role.

Created by: @totalf0otball / John
"""

import pandas as pd
import matplotlib.pyplot as plt
import matplotlib.patches as patches
import seaborn as sns
from pyfonts import load_google_font
import os
import re

# ---------------------------------------------------------
# 1. CONFIGURATION & SETUP
# ---------------------------------------------------------

LEAGUE_NAME = 'T5 European Leagues'
SEASON = '24-25'

# RELATIVE PATHS (GitHub Friendly)
DATA_FILE_PATH = os.path.join('data', 'goalkeepers.csv')
OUTPUT_DIR = 'outputs'

# Create output folder if it doesn't exist
os.makedirs(OUTPUT_DIR, exist_ok=True)

# FILTERS
MIN_90S = 9.0
LOW_SAMPLE_THRESHOLD = MIN_90S + 2  # 11.0 (The "+2 clear" rule)
MIN_PSXG = -0.1
MIN_PASS_CMP = 65
MIN_OPA = 0.7
AGE_LIMIT_YEAR = 1998 

# EXCLUSIONS (User can add specific names here to remove them from analysis)
EXCLUDED_PLAYERS = [] 

# OUTLIERS (User can add names here to highlight them in a different color)
OPA_OUTLIERS = [] 

# FONTS
try:
    font = load_google_font("Merriweather") 
    font_titlebold = load_google_font("Merriweather", weight="black")
    font_bold = load_google_font("Merriweather", weight="bold")
    font_italic = load_google_font("Merriweather", italic=True)
except Exception:
    # Fallback if internet/font is missing
    font = font_titlebold = font_bold = font_italic = 'sans-serif'

# OPTIONAL: adjustText
try:
    from adjustText import adjust_text
    HAS_ADJUST_TEXT = True
except ImportError:
    HAS_ADJUST_TEXT = False
    print("Tip: Install 'adjustText' via pip to prevent label overlap.")

# ---------------------------------------------------------
# 2. DATA LOADING & CLEANING
# ---------------------------------------------------------

if not os.path.exists(DATA_FILE_PATH):
    raise FileNotFoundError(f"Data file not found at: {DATA_FILE_PATH}. Please ensure 'goalkeepers.csv' is in the 'data' folder.")

data = pd.read_csv(DATA_FILE_PATH)
if data['Age'].isnull().any(): data.fillna({'Age': 0}, inplace=True)

df = data[
    (data['90s'] >= MIN_90S) &
    (data['PSxG+/-p90'] >= MIN_PSXG) &
    (data['Born'] >= AGE_LIMIT_YEAR) &
    (data['Total Pass Cmp%'] >= MIN_PASS_CMP) &
    (data['OPA p90'] >= MIN_OPA) &
    (~data['Player'].isin(EXCLUDED_PLAYERS))
].copy()

# ---------------------------------------------------------
# 3. METRICS
# ---------------------------------------------------------

# Cross Stopping
df['CrossStop_rate'] = df['Crosses stopped'] / df['Crosses faced'].replace(0, pd.NA)
df['CrossStop_rate'].fillna(0, inplace=True)
df['CrossStop_score'] = 0.5 * df['CrossStop_rate'].rank(pct=True) + 0.5 * df['Crosses stopped'].rank(pct=True)

# Profile Components
df['PSxG+/-_score'] = 0.6 * df['PSxG+/-p90'].rank(pct=True) + 0.4 * df['PSxG/SoT'].rank(pct=True)
df['PassCmp_score'] = (0.175 * df['Cmp Short Passes%'] + 0.175 * df['Cmp Med Passes%'] + 0.65 * df['Cmp Long Passes%']).rank(pct=True)
df['OPA_score'] = df['OPA p90'].rank(pct=True)

# Profiles
df['ShotStopper_Profile'] = 0.5 * df['PSxG+/-_score'] + 0.1 * df['PassCmp_score'] + 0.1 * df['OPA_score'] + 0.3 * df['CrossStop_score']
df['BallPlayer_Profile'] = 0.4 * df['PSxG+/-_score'] + 0.4 * df['PassCmp_score'] + 0.1 * df['OPA_score'] + 0.1 * df['CrossStop_score']
df['Sweeper_Profile'] = 0.4 * df['PSxG+/-_score'] + 0.1 * df['PassCmp_score'] + 0.3 * df['OPA_score'] + 0.2 * df['CrossStop_score']
df['Best_Profile_Score'] = df[['ShotStopper_Profile', 'BallPlayer_Profile', 'Sweeper_Profile']].mean(axis=1)

# ---------------------------------------------------------
# 4. PLOT 1: SCATTER (Story-Driven)
# ---------------------------------------------------------

print("Generating Plot 1: Performance Scatter...")

fig, ax = plt.subplots(figsize=(18, 12))
fig.patch.set_facecolor('#B3D8E0')
ax.set_facecolor('#daf0f5')

# Calculate Limits & Medians
x_median, y_median = df['PSxG+/-p90'].median(), df['Total Pass Cmp%'].median()
x_min, x_max = df['PSxG+/-p90'].min() - 0.05, df['PSxG+/-p90'].max() + 0.05
y_min, y_max = df['Total Pass Cmp%'].min() - 3, df['Total Pass Cmp%'].max() + 3
ax.set_xlim(x_min, x_max)
ax.set_ylim(y_min, y_max)

# --- 1. Story Quadrants ---
quad_colors = {'top_right': '#d1e7dd', 'top_left': '#cff4fc', 'bottom_right': '#fce5cd', 'bottom_left': '#f8d7da'}

# Top Right: Complete
ax.add_patch(patches.Rectangle((x_median, y_median), x_max, y_max, color=quad_colors['top_right'], alpha=0.3, zorder=0))
ax.text(x_max - 0.01, y_max - 0.5, "Complete Keepers\n(Elite Passing & Stopping)", 
        ha='right', va='top', fontsize=10, fontproperties=font_bold, color='#0f5132', alpha=0.7)

# Top Left: Ball Players
ax.add_patch(patches.Rectangle((x_min, y_median), x_median - x_min, y_max, color=quad_colors['top_left'], alpha=0.3, zorder=0))
ax.text(x_min + 0.01, y_max - 0.5, "Ball Playing Specialists\n(Good Feet, Lower Save Rate)", 
        ha='left', va='top', fontsize=10, fontproperties=font_bold, color='#055160', alpha=0.7)

# Bottom Right: Shot Stoppers
ax.add_patch(patches.Rectangle((x_median, y_min), x_max, y_median - y_min, color=quad_colors['bottom_right'], alpha=0.3, zorder=0))
ax.text(x_max - 0.01, y_min + 0.5, "Traditional Shot Stoppers\n(High Saves, Risky Passing)", 
        ha='right', va='bottom', fontsize=10, fontproperties=font_bold, color='#664d03', alpha=0.7)

# Bottom Left: Struggling
ax.add_patch(patches.Rectangle((x_min, y_min), x_median - x_min, y_median - y_min, color=quad_colors['bottom_left'], alpha=0.2, zorder=0))

# Spines
for spine in ax.spines.values(): spine.set_linewidth(3); spine.set_edgecolor('#0B2840')

# Split Data (Handle empty outlier lists gracefully)
df_outliers = df[df['Player'].isin(OPA_OUTLIERS)]
df_normal = df[~df['Player'].isin(OPA_OUTLIERS)]

# --- 2. Scatter Plots ---
sc = ax.scatter(
    df_normal['PSxG+/-p90'], df_normal['Total Pass Cmp%'],
    c=df_normal['OPA p90'], cmap='inferno', s=df_normal['CrossStop_score'] * 800,
    edgecolor='k', zorder=3, alpha=0.9
)

if not df_outliers.empty:
    ax.scatter(
        df_outliers['PSxG+/-p90'], df_outliers['Total Pass Cmp%'],
        color='deepskyblue', s=df_outliers['CrossStop_score'] * 800,
        edgecolor='k', zorder=4, alpha=0.9
    )

# Median Lines
ax.axvline(x=x_median, linestyle='--', color='#0B2840', alpha=0.5, zorder=2)
ax.axhline(y=y_median, linestyle='--', color='#0B2840', alpha=0.5, zorder=2)

# --- 3. Smart Labels ---
texts = []
for i, row in df.iterrows():
    display_name = f"{row['Player']}*" if row['90s'] <= LOW_SAMPLE_THRESHOLD else row['Player']
    t = ax.text(
        row['PSxG+/-p90'], row['Total Pass Cmp%']-0.5, display_name,
        fontsize=11, color='black', ha='center', va='center', fontproperties=font
    )
    texts.append(t)

if HAS_ADJUST_TEXT:
    adjust_text(texts, arrowprops=dict(arrowstyle='-', color='gray', lw=0.5), ax=ax)

# Legend/Colorbar
cbar = plt.colorbar(sc)
cbar.set_label('OPA p90 (Sweeper Actions)', fontsize=13, fontproperties=font)

# Descriptive Axes
ax.set_xlabel('Shot Stopping Ability (PSxG+/- per 90) $\longrightarrow$', fontsize=18, fontproperties=font_bold, color='#13344C')
ax.set_ylabel('Passing Reliability (Completion %) $\longrightarrow$', fontsize=18, fontproperties=font_bold, color='#13344C')

# Ticks
ax.tick_params(axis='both', which='major', width=2, length=8)
for label in ax.get_xticklabels() + ax.get_yticklabels():
    label.set_fontproperties(font_bold); label.set_fontsize(14); label.set_color('#0B2840')

# Titles
plt.text(0.071, 0.94, f'{LEAGUE_NAME} {SEASON} — Goalkeeper Performance', fontsize=30, fontproperties=font_titlebold, transform=fig.transFigure)
plt.text(0.071, 0.91, 'Created by @totalf0otball / John | Data via FBref', fontsize=18, fontproperties=font, transform=fig.transFigure)

# --- 4. DYNAMIC FILTER TEXT ---
filter_caption = (
    f"Filters: Min {int(MIN_90S)} matches. "
    f"PSxG+/- >= {MIN_PSXG}. "
    f"Pass Cmp >= {MIN_PASS_CMP}%. "
    f"OPA >= {MIN_OPA}."
)

plt.text(0.07, 0.04, filter_caption, fontsize=12, fontproperties=font_italic, transform=fig.transFigure)
plt.text(0.07, 0.02, 'Bubble Size = Cross Stopping | Color = OPA | Quadrants = Style Profile', fontsize=12, fontproperties=font_italic, transform=fig.transFigure)
plt.text(0.07, 0.00, f'(*) Denotes players within 2 games of the cutoff (≤ {int(LOW_SAMPLE_THRESHOLD)} 90s)', fontsize=12, fontproperties=font_italic, color='red', transform=fig.transFigure)

plt.tight_layout()
# Save Plot
scatter_path = os.path.join(OUTPUT_DIR, "Scatter_Story.png")
plt.savefig(scatter_path, dpi=250, bbox_inches='tight')
print(f"Saved: {scatter_path}")
plt.show()

# ---------------------------------------------------------
# 5. PLOT 2: HEATMAP
# ---------------------------------------------------------

print("Generating Plot 2: Profile Heatmap...")

top_n = 20
subset = df.nlargest(top_n, 'Best_Profile_Score')[['Player', 'ShotStopper_Profile', 'BallPlayer_Profile', 'Sweeper_Profile']]
subset.set_index('Player', inplace=True)

new_index_names = []
for player_name in subset.index:
    matches = df.loc[df['Player'] == player_name, '90s'].values[0]
    new_index_names.append(f"{player_name}*" if matches <= LOW_SAMPLE_THRESHOLD else player_name)
subset.index = new_index_names

subset.rename(columns={'ShotStopper_Profile': 'Shot Stopper', 'BallPlayer_Profile': 'Ball Player', 'Sweeper_Profile': 'Sweeper'}, inplace=True)

plt.figure(figsize=(10, 12)) 
ax = sns.heatmap(
    subset, annot=True, fmt=".2f", cmap='RdYlGn', 
    linewidths=1, linecolor='white',
    cbar_kws={'label': 'Percentile Rank Score', 'shrink': 0.8},
    annot_kws={"size": 11, "weight": "bold"}
)

ax.xaxis.tick_top(); ax.xaxis.set_label_position('top')
ax.tick_params(left=False, top=False)
plt.xlabel(""); plt.ylabel("")
plt.xticks(fontsize=12, fontweight='bold', fontproperties=font)
plt.yticks(fontsize=12, fontproperties=font)

title = f"{LEAGUE_NAME}\nGoalkeeper Profile Ratings ({SEASON})"
plt.suptitle(title, fontsize=18, fontweight='bold', fontproperties=font_titlebold, y=1)
plt.figtext(0.5, 0.935, "Created by @totalf0otball / John | Ranked by Aggregate Score", ha="center", fontsize=10, style='italic', fontproperties=font)
plt.figtext(0.5, 0.00, f"* Denotes players within 2 games of the cutoff (≤ {int(LOW_SAMPLE_THRESHOLD)} 90s)", ha="center", fontsize=10, style='italic', color='red', fontproperties=font)

plt.tight_layout()
# Save Plot
heatmap_path = os.path.join(OUTPUT_DIR, "Heatmap_Profile.png")
plt.savefig(heatmap_path, dpi=250, bbox_inches='tight')
print(f"Saved: {heatmap_path}")
plt.show()

print("Analysis Complete.")

import streamlit as st
import pandas as pd
import matplotlib.pyplot as plt
import matplotlib.patches as patches
import seaborn as sns
from pyfonts import load_google_font
import os

# --- PAGE CONFIGURATION ---
st.set_page_config(
    page_title="GK Analysis Tool",
    page_icon="🧤",
    layout="wide"
)

# --- FONTS ---
# (We handle fonts carefully for Streamlit since it runs on a server)
try:
    font = load_google_font("Merriweather") 
    font_titlebold = load_google_font("Merriweather", weight="black")
    font_bold = load_google_font("Merriweather", weight="bold")
    font_italic = load_google_font("Merriweather", italic=True)
except:
    font = font_titlebold = font_bold = font_italic = 'sans-serif'

# --- 1. DATA LOADING (Cached) ---
@st.cache_data
def load_data():
    # Looks for data in the 'data' folder
    file_path = os.path.join("data", "goalkeepers.csv")
    if not os.path.exists(file_path):
        return None
    data = pd.read_csv(file_path)
    if data['Age'].isnull().any(): 
        data.fillna({'Age': 0}, inplace=True)
    return data

# --- 2. CALCULATIONS ---
def calculate_metrics(df):
    # Cross Stopping
    df['CrossStop_rate'] = df['Crosses stopped'] / df['Crosses faced'].replace(0, pd.NA)
    df['CrossStop_rate'].fillna(0, inplace=True)
    df['CrossStop_score'] = 0.5 * df['CrossStop_rate'].rank(pct=True) + 0.5 * df['Crosses stopped'].rank(pct=True)

    # Profile Components
    df['PSxG+/-_score'] = 0.6 * df['PSxG+/-p90'].rank(pct=True) + 0.4 * df['PSxG/SoT'].rank(pct=True)
    df['PassCmp_score'] = (0.175 * df['Cmp Short Passes%'] + 0.175 * df['Cmp Med Passes%'] + 0.65 * df['Cmp Long Passes%']).rank(pct=True)
    df['OPA_score'] = df['OPA p90'].rank(pct=True)

    # Composite Profiles
    df['ShotStopper_Profile'] = 0.5 * df['PSxG+/-_score'] + 0.1 * df['PassCmp_score'] + 0.1 * df['OPA_score'] + 0.3 * df['CrossStop_score']
    df['BallPlayer_Profile'] = 0.4 * df['PSxG+/-_score'] + 0.4 * df['PassCmp_score'] + 0.1 * df['OPA_score'] + 0.1 * df['CrossStop_score']
    df['Sweeper_Profile'] = 0.4 * df['PSxG+/-_score'] + 0.1 * df['PassCmp_score'] + 0.3 * df['OPA_score'] + 0.2 * df['CrossStop_score']
    
    df['Best_Profile_Score'] = df[['ShotStopper_Profile', 'BallPlayer_Profile', 'Sweeper_Profile']].mean(axis=1)
    return df

# --- 3. PLOTTING FUNCTIONS ---
def plot_scatter(df, min_90s, league_name):
    # Create Figure
    fig, ax = plt.subplots(figsize=(16, 10))
    fig.patch.set_facecolor('#B3D8E0')
    ax.set_facecolor('#daf0f5')

    # Data Prep
    outliers = ['Wojciech Szczęsny', 'Iñaki Peña', 'Christos Mandas']
    df_normal = df[~df['Player'].isin(outliers)]
    df_outliers = df[df['Player'].isin(outliers)]

    # Plot
    sc = ax.scatter(
        df_normal['PSxG+/-p90'], df_normal['Total Pass Cmp%'],
        c=df_normal['OPA p90'], cmap='inferno', s=df_normal['CrossStop_score'] * 600,
        edgecolor='k', zorder=3, alpha=0.9
    )
    if not df_outliers.empty:
        ax.scatter(df_outliers['PSxG+/-p90'], df_outliers['Total Pass Cmp%'], color='deepskyblue', s=df_outliers['CrossStop_score'] * 600, edgecolor='k', zorder=4)

    # Reference Lines & Quadrants
    x_mid, y_mid = df['PSxG+/-p90'].median(), df['Total Pass Cmp%'].median()
    ax.axvline(x=x_mid, linestyle='--', color='#0B2840', alpha=0.5)
    ax.axhline(y=y_mid, linestyle='--', color='#0B2840', alpha=0.5)

    # Text
    try:
        from adjustText import adjust_text
        texts = []
        low_sample = min_90s + 2
        for i, row in df.iterrows():
            name = f"{row['Player']}*" if row['90s'] <= low_sample else row['Player']
            texts.append(ax.text(row['PSxG+/-p90'], row['Total Pass Cmp%'], name, fontsize=10, fontproperties=font))
        adjust_text(texts, arrowprops=dict(arrowstyle='-', color='gray', lw=0.5), ax=ax)
    except ImportError:
        pass # Graceful fallback if adjustText isn't installed

    # Style
    cbar = plt.colorbar(sc)
    cbar.set_label('OPA p90 (Sweeper Actions)', fontsize=12, fontproperties=font)
    ax.set_xlabel('Shot Stopping (PSxG+/- per 90)', fontsize=14, fontproperties=font_bold)
    ax.set_ylabel('Pass Completion %', fontsize=14, fontproperties=font_bold)
    
    plt.text(0.05, 0.95, f"{league_name} — Performance", fontsize=24, fontproperties=font_titlebold, transform=fig.transFigure)
    
    return fig

def plot_heatmap(df, league_name):
    # Prepare Data
    top_n = 20
    subset = df.nlargest(top_n, 'Best_Profile_Score')[['Player', 'ShotStopper_Profile', 'BallPlayer_Profile', 'Sweeper_Profile']]
    subset.set_index('Player', inplace=True)
    subset.rename(columns={'ShotStopper_Profile': 'Shot Stopper', 'BallPlayer_Profile': 'Ball Player', 'Sweeper_Profile': 'Sweeper'}, inplace=True)

    # Plot
    fig, ax = plt.subplots(figsize=(10, 12))
    sns.heatmap(subset, annot=True, fmt=".2f", cmap='RdYlGn', linewidths=1, linecolor='white', cbar_kws={'shrink': 0.8})
    
    ax.xaxis.tick_top()
    plt.title(f"Top 20 Profile Ratings ({league_name})", fontsize=18, fontproperties=font_titlebold, y=1.05)
    return fig

# --- 4. MAIN APP LAYOUT ---
def main():
    st.title("🧤 Goalkeeper Scouting Dashboard")
    st.markdown("Interact with the sliders to filter players and generate updated visualizations.")

    # Load Data
    raw_data = load_data()
    if raw_data is None:
        st.error("Data file not found! Please ensure 'data/goalkeepers.csv' exists.")
        return

    # --- SIDEBAR FILTERS ---
    st.sidebar.header("Filter Settings")
    
    league_name = st.sidebar.text_input("League Name", "T5 European Leagues")
    min_90s = st.sidebar.slider("Minimum 90s Played", 5.0, 30.0, 9.0, 0.5)
    min_psxg = st.sidebar.slider("Min PSxG+/- per 90", -0.5, 0.5, -0.1, 0.05)
    min_pass = st.sidebar.slider("Min Pass Completion %", 50, 90, 65, 1)
    min_opa = st.sidebar.slider("Min OPA per 90", 0.0, 2.0, 0.7, 0.1)
    age_limit = st.sidebar.number_input("Born After (Year)", 1980, 2010, 1998)

    # --- FILTERING LOGIC ---
    df_filtered = raw_data[
        (raw_data['90s'] >= min_90s) &
        (raw_data['PSxG+/-p90'] >= min_psxg) &
        (raw_data['Born'] >= age_limit) &
        (raw_data['Total Pass Cmp%'] >= min_pass) &
        (raw_data['OPA p90'] >= min_opa)
    ].copy()

    # Calculate metrics on the filtered dataset
    if not df_filtered.empty:
        df_final = calculate_metrics(df_filtered)
        
        # Display Player Count
        st.sidebar.markdown(f"**Players Found:** {len(df_final)}")
        
        # --- TABS FOR VISUALS ---
        tab1, tab2, tab3 = st.tabs(["📊 Scatter Plot", "🔥 Heatmap", "📋 Data Table"])

        with tab1:
            st.subheader("Performance Matrix")
            fig_scatter = plot_scatter(df_final, min_90s, league_name)
            st.pyplot(fig_scatter)
            
            # Download Button
            fn = "scatter.png"
            fig_scatter.savefig(fn, dpi=300, bbox_inches='tight')
            with open(fn, "rb") as img:
                st.download_button(label="Download Scatter Plot", data=img, file_name=fn, mime="image/png")

        with tab2:
            st.subheader("Profile Ratings Leaderboard")
            if len(df_final) > 0:
                fig_heat = plot_heatmap(df_final, league_name)
                st.pyplot(fig_heat)
                
                # Download Button
                fn_heat = "heatmap.png"
                fig_heat.savefig(fn_heat, dpi=300, bbox_inches='tight')
                with open(fn_heat, "rb") as img:
                    st.download_button(label="Download Heatmap", data=img, file_name=fn_heat, mime="image/png")
            else:
                st.warning("Not enough players to generate a heatmap.")

        with tab3:
            st.subheader("Raw Data")
            st.dataframe(df_final[['Player', 'Squad', 'Age', '90s', 'PSxG+/-p90', 'Total Pass Cmp%', 'OPA p90']])

    else:
        st.warning("No players match these filters. Try lowering the thresholds in the sidebar.")

if __name__ == "__main__":
    main()

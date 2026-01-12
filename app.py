import streamlit as st
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
import joblib
from sklearn.metrics import confusion_matrix

# ================= PAGE CONFIG =================
st.set_page_config(
    page_title="Netflix Analytics & Prediction Dashboard",
    page_icon="🎬",
    layout="wide"
)

# ================= DARK MODE =================
st.markdown("""
<style>
body { background-color: #0e0e0e; color: white; }
[data-testid="stMetricLabel"] { color: #b084ff; }
[data-testid="stMetricValue"] { color: white; }
</style>
""", unsafe_allow_html=True)

st.title("🎬 Netflix Movies & TV Shows Dashboard")
st.caption("EDA + Machine Learning (Movie vs TV Show Prediction)")

# ================= HELPER =================
def parse_duration(x):
    if isinstance(x, str):
        if "min" in x:
            return int(x.replace(" min", ""))
        if "Season" in x:
            return int(x.split(" ")[0]) * 60
    return None

# ================= LOAD DATA =================
df = pd.read_csv("netflix_titles.csv")

df["country"].fillna("Unknown", inplace=True)
df["rating"].fillna("Not Rated", inplace=True)
df["listed_in"].fillna("Unknown", inplace=True)

# ================= GENRE EXPLODE =================
genre_df = df.copy()
genre_df["listed_in"] = genre_df["listed_in"].str.split(", ")
genre_df = genre_df.explode("listed_in")

# ================= SIDEBAR FILTERS =================
st.sidebar.header("🔍 Filters")

type_filter = st.sidebar.multiselect(
    "Content Type",
    df["type"].unique(),
    default=df["type"].unique()
)

genre_filter = st.sidebar.multiselect(
    "Genre",
    sorted(genre_df["listed_in"].unique())
)

year_filter = st.sidebar.slider(
    "Release Year",
    int(df["release_year"].min()),
    int(df["release_year"].max()),
    (2000, int(df["release_year"].max()))
)

filtered_df = genre_df[
    (genre_df["type"].isin(type_filter)) &
    (genre_df["release_year"].between(year_filter[0], year_filter[1]))
]

if genre_filter:
    filtered_df = filtered_df[filtered_df["listed_in"].isin(genre_filter)]

# ================= TABS =================
tab1, tab2, tab3 = st.tabs(["📊 Dashboard", "🤖 Prediction", "📘 Insights"])

# ======================================================
# ======================= DASHBOARD ====================
# ======================================================
with tab1:

    # -------- KPIs --------
    st.markdown("## 📌 Key Metrics")
    c1, c2, c3, c4 = st.columns(4)

    c1.metric("Total Titles", len(filtered_df))
    c2.metric("Movies", len(filtered_df[filtered_df["type"] == "Movie"]))
    c3.metric("TV Shows", len(filtered_df[filtered_df["type"] == "TV Show"]))
    c4.metric("Unique Genres", filtered_df["listed_in"].nunique())

    # -------- CHART GRID --------
    st.markdown("## 📊 Content Distribution")
    g1, g2, g3, g4 = st.columns(4)

    # Movies vs TV
    with g1:
        counts = filtered_df["type"].value_counts()
        fig, ax = plt.subplots(figsize=(3, 3))
        ax.bar(counts.index, counts.values, color="black", edgecolor="purple", linewidth=2)
        ax.set_title("Movies vs TV", color="purple", fontsize=10)
        st.pyplot(fig)

    # Pie
    with g2:
        fig, ax = plt.subplots(figsize=(3, 3))
        ax.pie(
            counts.values,
            labels=counts.index,
            autopct="%1.1f%%",
            colors=["black", "purple"],
            textprops={"color": "white"}
        )
        ax.set_title("Content Share", color="purple", fontsize=10)
        st.pyplot(fig)

    # Genres
    with g3:
        top_genres = filtered_df["listed_in"].value_counts().head(8)
        fig, ax = plt.subplots(figsize=(3, 3))
        ax.barh(top_genres.index, top_genres.values, color="black", edgecolor="purple", linewidth=2)
        ax.invert_yaxis()
        ax.set_title("Top Genres", color="purple", fontsize=10)
        st.pyplot(fig)

    # Countries
    with g4:
        country_df = filtered_df.copy()
        country_df["country"] = country_df["country"].str.split(", ")
        country_df = country_df.explode("country")
        top_countries = country_df["country"].value_counts().head(8)

        fig, ax = plt.subplots(figsize=(3, 3))
        ax.barh(top_countries.index, top_countries.values, color="black", edgecolor="purple", linewidth=2)
        ax.invert_yaxis()
        ax.set_title("Top Countries", color="purple", fontsize=10)
        st.pyplot(fig)

    # -------- TIME TRENDS --------
    st.markdown("## 📅 Trends Over Time")
    t1, t2 = st.columns(2)

    with t1:
        yearly = filtered_df["release_year"].value_counts().sort_index()
        fig, ax = plt.subplots(figsize=(4, 3))
        ax.plot(yearly.index, yearly.values, color="purple", linewidth=2)
        ax.set_title("Titles per Year", color="purple")
        st.pyplot(fig)

    with t2:
        trend = filtered_df.groupby(["release_year", "type"]).size().unstack().fillna(0)
        fig, ax = plt.subplots(figsize=(4, 3))
        trend.plot(ax=ax, color=["black", "purple"])
        ax.set_title("Movie vs TV Trend", color="purple")
        st.pyplot(fig)

    # -------- TABLE --------
    st.markdown("## 📋 Filtered Dataset")
    st.dataframe(
        filtered_df[["type", "title", "country", "release_year", "rating", "listed_in"]],
        use_container_width=True
    )

    st.download_button(
        "📥 Download Filtered Data",
        filtered_df.to_csv(index=False),
        "filtered_netflix_data.csv"
    )

# ======================================================
# ======================= PREDICTION ===================
# ======================================================
with tab2:
    st.markdown("## 🤖 Predict Content Type")

    model = joblib.load("netflix_type_model.pkl")
    encoder = joblib.load("rating_encoder.pkl")

    p1, p2, p3 = st.columns(3)

    with p1:
        year = st.number_input("Release Year", 1950, 2025, 2020)

    with p2:
        rating = st.selectbox("Rating", encoder.classes_)

    with p3:
        duration = st.number_input("Duration (minutes or seasons×60)", 30, 600, 120)

    if st.button("🔮 Predict"):
        rating_enc = encoder.transform([rating])[0]
        pred = model.predict([[year, rating_enc, duration]])[0]
        st.success("🎬 Movie" if pred == 0 else "📺 TV Show")

    # -------- MODEL EVALUATION (SAFE) --------
    st.markdown("### 📊 Model Evaluation")

    eval_df = df[["type", "rating", "release_year", "duration"]].dropna()
    eval_df = eval_df[eval_df["rating"].isin(encoder.classes_)]

    eval_df["rating_encoded"] = encoder.transform(eval_df["rating"])
    eval_df["duration_num"] = eval_df["duration"].apply(parse_duration)
    eval_df = eval_df.dropna()

    X_eval = eval_df[["release_year", "rating_encoded", "duration_num"]]
    y_true = eval_df["type"].map({"Movie": 0, "TV Show": 1})

    y_pred = model.predict(X_eval)

    cm = confusion_matrix(y_true, y_pred)

    fig, ax = plt.subplots(figsize=(4, 3))
    sns.heatmap(cm, annot=True, fmt="d", cmap="Purples", ax=ax)
    ax.set_xlabel("Predicted")
    ax.set_ylabel("Actual")
    st.pyplot(fig)

    # Feature importance
    st.markdown("### 🔍 Feature Importance")
    coef_df = pd.DataFrame({
        "Feature": ["Release Year", "Rating", "Duration"],
        "Importance": model.coef_[0]
    })
    st.dataframe(coef_df)

# ======================================================
# ======================= INSIGHTS =====================
# ======================================================
with tab3:
    st.markdown("## 💡 Business Insights")
    st.markdown("""
- Netflix hosts **more Movies than TV Shows**, but TV content has grown rapidly post-2016  
- **Drama & International genres** dominate the catalog  
- The US and India are **top content producers**  
- Content growth accelerated significantly after 2015
""")

    st.markdown("## ⚠️ Data Limitations")
    st.markdown("""
- Dataset lacks viewership, revenue, and user behavior  
- Predictions rely only on metadata  
- Model is for educational and analytical purposes
""")

    st.markdown("## 🧾 Tech Stack")
    st.markdown("""
- Python, Pandas, NumPy  
- Matplotlib, Seaborn  
- Streamlit  
- Scikit-learn
""")

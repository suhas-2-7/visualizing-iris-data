import streamlit as st
import pandas as pd
import seaborn as sns
import plotly.express as px
import matplotlib.pyplot as plt
from streamlit_lottie import st_lottie
import requests

# Load Lottie animation
def load_lottie_url(url):
    r = requests.get(url)
    if r.status_code != 200:
        return None
    return r.json()

# Load data
@st.cache_data
def load_data():
    return sns.load_dataset("iris")

df = load_data()
species_list = df['species'].unique()

# Sidebar
st.sidebar.title("⚙️ Filters")
selected_species = st.sidebar.multiselect("Choose species to display", species_list, default=species_list)

filtered_df = df[df['species'].isin(selected_species)]

# --- Lottie animation at top ---
lottie_animation = load_lottie_url("https://assets2.lottiefiles.com/packages/lf20_kkflmtur.json")
st_lottie(lottie_animation, height=200, key="intro_anim")

# Title
st.title("🌼 Interactive Iris EDA Dashboard")

st.markdown("""
Welcome to a powerful, interactive EDA dashboard built with **Streamlit**, **Plotly**, and **Seaborn**.  
Explore the Iris dataset visually with filters, plots, and animations.

---

### 🌸 What’s the Iris Dataset?

- 150 flower samples
- 3 species: *Setosa*, *Versicolor*, *Virginica*
- 4 features: Sepal & Petal (length + width)
""")

# Dataset overview
with st.expander("📄 View Dataset Description & Table"):
    st.markdown("""
    **Features:**
    - `sepal_length`: Sepal length (cm)
    - `sepal_width`: Sepal width (cm)
    - `petal_length`: Petal length (cm)
    - `petal_width`: Petal width (cm)
    - `species`: Type of iris flower

    **Applications:**
    - Pattern recognition
    - Model training for classification
    - Feature engineering practice
    """)
    st.dataframe(filtered_df)

# Summary statistics
st.subheader("📈 Summary Statistics")
st.dataframe(filtered_df.describe(), use_container_width=True)

# Tabs for visualizations
tab1, tab2, tab3, tab4, tab5 = st.tabs(["📊 Histogram", "📦 Box Plot", "🌈 Scatter (2D)", "🔍 Pairwise", "🔥 Correlation Heatmap"])

# Histogram
with tab1:
    st.markdown("### 🌻 Histogram with KDE")
    feature = st.selectbox("Feature", df.columns[:-1], key="hist")
    fig = px.histogram(filtered_df, x=feature, color="species", marginal="rug", nbins=20,
                       title=f"Distribution of {feature}", opacity=0.75)
    st.plotly_chart(fig)

# Box plot
with tab2:
    st.markdown("### 📦 Box Plot")
    feature = st.selectbox("Feature", df.columns[:-1], key="box")
    fig = px.box(filtered_df, x="species", y=feature, color="species", points="all",
                 title=f"Box Plot of {feature} by Species")
    st.plotly_chart(fig)

# 2D scatter
with tab3:
    st.markdown("### ✨ Scatter Plot (2D)")
    x = st.selectbox("X-axis", df.columns[:-1], key="xscatter")
    y = st.selectbox("Y-axis", df.columns[:-1], key="yscatter")
    fig = px.scatter(filtered_df, x=x, y=y, color="species", symbol="species",
                     title=f"{x} vs {y} Scatter Plot")
    st.plotly_chart(fig)

# Pairplot
with tab4:
    st.markdown("### 🧠 Seaborn Pairplot (all features)")
    st.info("This uses Seaborn and is not interactive, but shows great feature relationships.")
    fig = sns.pairplot(filtered_df, hue="species")
    st.pyplot(fig)

# Heatmap
with tab5:
    st.markdown("### 🔥 Correlation Heatmap")
    fig, ax = plt.subplots()
    sns.heatmap(filtered_df.drop("species", axis=1).corr(), annot=True, cmap="coolwarm", ax=ax)
    st.pyplot(fig)

# Footer
st.markdown("---")
st.markdown("""
### ✅ About This App
- Built with ❤️ using **Streamlit**, **Plotly**, **Seaborn**
- Fully interactive EDA for Iris Dataset
- Designed for beginners & portfolios

### 📢 Impress Interviewers:
This app shows you can:
- Clean and analyze real datasets
- Build dashboards from scratch
- Make analytics interactive and attractive!

---
""")

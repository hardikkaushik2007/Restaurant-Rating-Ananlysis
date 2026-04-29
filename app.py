# =========================
# IMPORT LIBRARIES
# =========================

import streamlit as st
import pandas as pd
import numpy as np
import joblib

import plotly.express as px
import plotly.graph_objects as go

# ==========================================================
# PAGE CONFIG
# ==========================================================

st.set_page_config(
    page_title="Zomato Analytics Dashboard",
    page_icon="🍽️",
    layout="wide"
)

# ==========================================================
# CUSTOM CSS
# ==========================================================

st.markdown("""
<style>

/* ==========================================================
MAIN BACKGROUND
========================================================== */

.stApp {

    background: linear-gradient(
        to right,
        #0f172a,
        #1e293b
    );

    color: white;
}

/* ==========================================================
TITLE COLORS
========================================================== */

h1, h2, h3 {

    color: #f97316 !important;
}

/* ==========================================================
SIDEBAR
========================================================== */

section[data-testid="stSidebar"] {

    background-color: #111827;
}

/* ==========================================================
METRIC CARDS
========================================================== */

div[data-testid="metric-container"] {

    background-color: #1e293b;

    border: 1px solid #334155;

    padding: 15px;

    border-radius: 15px;
}

/* ==========================================================
BUTTON
========================================================== */

.stButton button {

    background-color: #f97316;

    color: white;

    border-radius: 10px;

    border: none;

    height: 50px;

    width: 100%;

    font-size: 18px;
}

/* ==========================================================
SELECT BOX
========================================================== */

div[data-baseweb="select"] {

    color: black;
}

/* ==========================================================
SLIDER
========================================================== */

.stSlider {

    padding-top: 10px;
}

/* ==========================================================
SUCCESS MESSAGE
========================================================== */

.stSuccess {

    background-color: #14532d;
}

/* ==========================================================
DATAFRAME
========================================================== */

[data-testid="stDataFrame"] {

    border-radius: 15px;
}

</style>
""", unsafe_allow_html=True)

# ==========================================================
# LOAD DATASET
# ==========================================================

@st.cache_data
def load_data():

    df = pd.read_csv(
        "data/processed/zomato_cleaned.csv"
    )

    return df

df = load_data()

# ==========================================================
# LOAD MODEL
# ==========================================================

model = joblib.load(
    "models/zomato_rating_model.pkl"
)

# ==========================================================
# SIDEBAR
# ==========================================================

st.sidebar.title("🍴 Navigation")

page = st.sidebar.radio(
    "Go To",
    [
        "Home",
        "EDA Dashboard",
        "Prediction",
        "Insights"
    ]
)

# ==========================================================
# HOME PAGE
# ==========================================================

if page == "Home":

    st.title("🍽️ Zomato Restaurant Analytics Dashboard")

    st.markdown("""
    ## AI + Data Analytics Project

    This dashboard includes:

    - Interactive Data Visualization
    - Restaurant Analysis
    - Business Insights
    - ML Rating Prediction
    - Dynamic Charts
    """)

    col1, col2, col3 = st.columns(3)

    with col1:

        st.metric(
            "Total Restaurants",
            df.shape[0]
        )

    with col2:

        st.metric(
            "Average Rating",
            round(df['rate'].mean(), 2)
        )

    with col3:

        st.metric(
            "Average Cost",
            round(
                df['approx_cost(for two people)'].mean(),
                2
            )
        )

    st.markdown("---")

    st.subheader("📌 Dataset Preview")

    st.dataframe(df.head(10))

# ==========================================================
# EDA DASHBOARD
# ==========================================================

elif page == "EDA Dashboard":

    st.title("📊 Interactive EDA Dashboard")

    # =========================
    # LOCATION FILTER
    # =========================

    locations = sorted(df['location'].unique())

    selected_location = st.selectbox(
        "Select Location",
        locations
    )

    filtered_df = df[
        df['location'] == selected_location
    ]

    # =========================
    # KPI SECTION
    # =========================

    col1, col2, col3 = st.columns(3)

    with col1:

        st.metric(
            "Restaurants",
            filtered_df.shape[0]
        )

    with col2:

        st.metric(
            "Average Rating",
            round(filtered_df['rate'].mean(), 2)
        )

    with col3:

        st.metric(
            "Average Votes",
            round(filtered_df['votes'].mean(), 2)
        )

    # =========================
    # RATING DISTRIBUTION
    # =========================

    fig1 = px.histogram(
        filtered_df,
        x='rate',
        nbins=20,
        title='Restaurant Rating Distribution'
    )

    st.plotly_chart(
        fig1,
        use_container_width=True
    )

    # =========================
    # ONLINE ORDER ANALYSIS
    # =========================

    fig2 = px.pie(
        filtered_df,
        names='online_order',
        title='Online Order Availability'
    )

    st.plotly_chart(
        fig2,
        use_container_width=True
    )

    # =========================
    # TOP CUISINES
    # =========================

    top_cuisines = (
        filtered_df['cuisines']
        .value_counts()
        .head(10)
    )

    fig3 = px.bar(
        x=top_cuisines.index,
        y=top_cuisines.values,
        title='Top 10 Cuisines'
    )

    st.plotly_chart(
        fig3,
        use_container_width=True
    )

    # =========================
    # COST VS RATING
    # =========================

    fig4 = px.scatter(
        filtered_df,
        x='approx_cost(for two people)',
        y='rate',
        color='online_order',
        title='Cost vs Rating'
    )

    st.plotly_chart(
        fig4,
        use_container_width=True
    )

# ==========================================================
# PREDICTION PAGE
# ==========================================================

elif page == "Prediction":

    st.title("⭐ Restaurant Rating Prediction")

    votes = st.slider(
        "Votes",
        0,
        50000,
        1000
    )

    cost = st.slider(
        "Approx Cost For Two",
        100,
        5000,
        500
    )

    online_order = st.selectbox(
        "Online Order",
        ['Yes', 'No']
    )

    book_table = st.selectbox(
        "Book Table",
        ['Yes', 'No']
    )

    # =========================
    # ENCODING
    # =========================

    online_order = (
        1 if online_order == 'Yes'
        else 0
    )

    book_table = (
        1 if book_table == 'Yes'
        else 0
    )

    # =========================
    # SAMPLE INPUT
    # =========================

    sample_input = pd.DataFrame({

        'online_order': [online_order],

        'book_table': [book_table],

        'votes': [votes],

        'location': [0],

        'rest_type': [0],

        'cuisines': [0],

        'approx_cost(for two people)': [cost],

        'listed_in(type)': [0]

    })

    # =========================
    # PREDICT BUTTON
    # =========================

    if st.button("Predict Rating"):

        prediction = model.predict(
            sample_input
        )

        st.success(
            f"Predicted Restaurant Rating: {round(prediction[0],2)} ⭐"
        )

# ==========================================================
# INSIGHTS PAGE
# ==========================================================

elif page == "Insights":

    st.title("📈 Business Insights")

    st.markdown("""
    ## Key Insights

    ### 1. Mid-range restaurants perform best.

    Restaurants with moderate pricing generally
    receive better engagement and ratings.

    ---

    ### 2. Online ordering improves visibility.

    Restaurants offering online delivery
    often gain higher customer reach.

    ---

    ### 3. Votes strongly influence ratings.

    Restaurants with more votes generally
    have more reliable ratings.

    ---

    ### 4. Popular locations dominate restaurant density.

    Areas with higher footfall contain
    more restaurant listings.

    ---

    ### 5. Certain cuisines dominate demand.

    North Indian, Chinese and Fast Food
    cuisines appear most frequently.
    """)
import streamlit as st
import pandas as pd
import matplotlib.pyplot as plt
import random
import altair as alt

# Code below to be moved to a new folder to modularize it
def stats(df):
    st.header("General Statistics")
    st.write(df.describe())

def exchange(df):
    df['date'] = pd.to_datetime(df['date'])
        
    # allow user to select currency to plot
    currencies = df['currency'].unique().tolist()
    selected = st.multiselect("Select currencies", currencies, default=currencies[0]) # only select first currency if user choose multiple

    df_filtered = df[df['currency'].isin(selected)]

    # altair chart
    chart = alt.Chart(df_filtered).mark_line().encode(
        x='date:T',
        y=alt.Y('exchange_rate:Q', scale=alt.Scale(domain=[1.2, 1.8])),
        color='currency:N'
    ).properties(
        width=900,
        height=400,
        title="Currency Exchange Rate Over Time"
    )

    st.altair_chart(chart, use_container_width=True)

def emotions(df):
    st.header("Emotions")
    emotion_counts = df['emotions'].value_counts()
    fig, ax = plt.subplots(figsize=(1.5, 1.5))
    emotion_counts.plot(kind='pie', autopct='%1.1f%%', startangle=140, ax=ax, textprops={'fontsize': 5})
    #ax.set_title("Emotion distribution")
    ax.set_ylabel("")  
    plt.tight_layout(pad=0.5)
    st.pyplot(fig)


st.title("DataHackFest 2025")
st.text("This is a forex rate predictor with news sentiment analysis")

st.sidebar.title("Nav")

uploaded_file = st.sidebar.file_uploader("Upload a custom csv file with your personal music emotions", type=["csv"])

options = st.sidebar.radio("View the dataset:", ("Home", "General statistics", "Exchange Rates", "Emotions"))

# datasets and page setup
if uploaded_file:
    df = pd.read_csv(uploaded_file)
else:
    df = pd.read_csv("data/forex/filtered_forex_rates.csv")

if options == "General statistics":
    stats(df)
elif options == "Exchange Rates":
    exchange(df)
elif options == "Emotions":
    emotions(df)
import streamlit as st
import pandas as pd
import matplotlib.pyplot as plt
import random

# Code below to be moved to a new folder to modularize it
def stats(df):
    st.header("General Statistics")
    st.write(df.describe())

def genres(df):
    st.header("Genres")
    fix, ax = plt.subplots(1, 1)
    ax.scatter(df['genres'], df['emotions'], alpha=0.5)
    ax.set_xlabel('Genres')
    ax.set_ylabel('Emotions')
    st.pyplot(fix)

def emotions(df):
    st.header("Emotions")
    emotion_counts = df['emotions'].value_counts()
    fig, ax = plt.subplots(figsize=(1.5, 1.5))
    emotion_counts.plot(kind='pie', autopct='%1.1f%%', startangle=140, ax=ax, textprops={'fontsize': 5})
    #ax.set_title("Emotion distribution")
    ax.set_ylabel("")  
    plt.tight_layout(pad=0.5)
    st.pyplot(fig)


st.title("Data HackFest 2025")
st.text("This is a emotion based music playlist generator app with multiple customization options.")

st.sidebar.title("Nav")

uploaded_file = st.sidebar.file_uploader("Upload a custom csv file with your personal music emotions", type=["csv"])

options = st.sidebar.radio("View the dataset:", ("Home", "General statistics", "Genres", "Emotions"))

# datasets and page setup
if uploaded_file:
    df = pd.read_csv(uploaded_file)
else:
    df = pd.read_csv("data/music.csv")

if options == "General statistics":
    stats(df)
elif options == "Genres":
    genres(df)
elif options == "Emotions":
    emotions(df)

# user input
user_input = st.text_input("Enter your vibe today", placeholder="Type the emotion here...")

if user_input:
    st.write(f"Great! Today's vibe is - {user_input}!")

output_bool = random.choice([True, False])  # temp model prediction

while not output_bool:
    st.write("Generating your playlist... Please wait and explore the dataset meanwhile")  
    output_bool = random.choice([True, False])   
    if output_bool:
        break

#st.experimental_rerun()
st.write("Your playlist is generated successfully!")
import pandas as pd
import json
import matplotlib.pyplot as plt

with open('data/news/daily_scores.json', 'r') as file:
    sentiment_scores = json.load(file)

df_sentiment = pd.DataFrame(list(sentiment_scores.items()), columns=['jsonDate', 'sentiment'])
df_sentiment['date'] = pd.to_datetime(df_sentiment['jsonDate'], format='%Y-%m-%d')
# print(df_sentiment.head())

forex_file = 'data/forex/cad_filtered_forex_rates.csv'
df_forex = pd.read_csv(forex_file)

df_forex['date'] = pd.to_datetime(df_forex['date'], format='%Y-%m-%d')
df_merged = pd.merge(df_forex, df_sentiment, on='date', how='left')
df_sentimentColumns = pd.DataFrame(df_merged['sentiment'].apply(pd.Series))

df_sentimentColumns.rename(columns={'negative': 'sentiment_negative', 
                                    'positive': 'sentiment_positive'}, inplace=True)

df_merged = pd.concat([df_merged.drop(columns='sentiment'), df_sentimentColumns], axis=1)
#df_merged = pd.concat([df_merged, df_sentimentColumns], axis=1)

df_merged = df_merged.drop(columns=['jsonDate', 'base_currency', 'currency_name', 'currency'])
print(df_merged.head())

df_merged = df_merged.sort_values(by='date', ascending=True)
df_merged.to_csv('data/merged.csv', index=False)
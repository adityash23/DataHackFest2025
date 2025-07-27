# extract the forex values only in the specified date range
import pandas as pd

df = pd.read_csv("data/forex/daily_forex_rates.csv") 

# make the 'date' column a datetime object
df['date'] = pd.to_datetime(df['date'], format='%Y-%m-%d', errors='coerce')

# dates are inclusive
start_date = '2015-01-01'
end_date = '2021-12-31'

filtered_df = df[(df['date'] >= start_date) & (df['date'] <= end_date)]

filtered_df.to_csv("filtered_forex_rates.csv", index=False)

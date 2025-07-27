# extract CAD forex rates from dataset with multiple currencies
import pandas as pd

df = pd.read_csv("data/forex/filtered_forex_rates.csv") 

df_usd = df[df['currency'].str.contains("CAD", na=False)]

df_usd.to_csv("data/forex/cad_filtered_forex_rates.csv", index=False)
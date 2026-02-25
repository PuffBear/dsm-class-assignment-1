# Task 1: Data Quality — Drop unreliable columns and produce a clean CSV for downstream analysis.

import pandas as pd

# Load the raw scraped output from the GDACS scraper
df = pd.read_csv('./scraping/gdacs_earthquake_data.csv')

# Drop 'social_media_note' — column was consistently empty/unreliable across all scraped events
df_clean = df.drop('social_media_note', axis=1)

# Persist the cleaned dataset for use in Task 2 and Task 3/4 pipelines
df_clean.to_csv('./scraping/gdacs_earthquake_data_cleaned.csv', index=False)

# Sanity check: print remaining null counts per column to verify data completeness
print(df_clean.isna().sum())
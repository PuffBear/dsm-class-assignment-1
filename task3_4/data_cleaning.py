import pandas as pd
import os

def parse_population(pop_str):
    if pd.isna(pop_str) or not isinstance(pop_str, str):
        return pop_str
    pop_str = pop_str.lower().replace(',', '')
    if 'thousand' in pop_str:
        return float(pop_str.replace('thousand', '').strip()) * 1000
    if 'million' in pop_str:
        return float(pop_str.replace('million', '').strip()) * 1000000
    try:
        return float(pop_str)
    except:
        return 0

def clean_data():
    input_file = "/Users/Agriya/Desktop/spring26/dsm/Assignment 1/task1/gdacs_earthquake_data_cleaned.csv"
    output_file = "/Users/Agriya/Desktop/spring26/dsm/Assignment 1/task3_4/cleaned_dataset.csv"
    
    if not os.path.exists(input_file):
        print(f"Error: {input_file} not found.")
        return
        
    df = pd.read_csv(input_file)
    
    # 1. Clean exposed population
    df['exposed_population_mmi_clean'] = df['exposed_population_mmi'].apply(parse_population)
    
    # 2. Calculate Forgotten Crisis Index (News Volume / Population Impact)
    df['forgotten_crisis_index'] = df['total_articles'] / df['exposed_population_mmi_clean'].replace(0, 1)
    
    # 3. Clean coping capacity (keep only the float value)
    # E.g. "4.2 (Philippines)" -> 4.2
    df['coping_capacity_clean'] = df['inform_coping_capacity'].str.extract(r'([\d\.]+)').astype(float)
    
    # 4. Clean magnitude (remove 'M')
    df['magnitude_clean'] = df['magnitude'].str.extract(r'([\d\.]+)').astype(float)
    
    # Parse timeline data to JSON for easier consumption by Streamlit if needed
    # Or keep it as is. We'll leave it as is.
    
    # Save the cleaned dataset
    df.to_csv(output_file, index=False)
    print(f"Cleaned data successfully saved to {output_file}")

if __name__ == "__main__":
    clean_data()

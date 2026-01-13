"""Separate dataset into category-specific CSV files.

This module segments the annotated dataset into individual CSV files
for each content category to enable targeted analysis.
"""

import pandas as pd
import os
from pathlib import Path

def separate_data_by_category():
    print("Reading data_annotated.csv...")
    df = pd.read_csv('data_annotated.csv')
    
    data_dir = Path('data')
    data_dir.mkdir(exist_ok=True)
    
    categories = df['Categories'].unique()
    print(f"Found {len(categories)} categories:")
    for cat in categories:
        print(f"  - {cat}")
    
    for category in categories:
        category_data = df[df['Categories'] == category]
        
        safe_filename = category.replace('&', 'and').replace('/', '_').replace(' ', '_').lower()
        filename = f"{safe_filename}.csv"
        filepath = data_dir / filename
        
        category_data.to_csv(filepath, index=False)
        
        print(f"Saved {len(category_data)} entries for '{category}' to {filepath}")
    
    print(f"\nAll files saved successfully in the 'data' folder!")
    
    print("\nSummary:")
    total_entries = len(df)
    print(f"Total entries in original file: {total_entries}")
    
    verification_count = 0
    for category in categories:
        count = len(df[df['Categories'] == category])
        verification_count += count
        print(f"  {category}: {count} entries")
    
    print(f"Verification: {verification_count} entries distributed across all files")

if __name__ == "__main__":
    try:
        separate_data_by_category()
        print("Script completed successfully!")
    except Exception as e:
        print(f"Error occurred: {e}")
        import traceback
        traceback.print_exc()
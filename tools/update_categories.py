"""
Update category names in annotated datasets to ensure consistency.

This module standardizes category naming conventions across datasets by applying
predefined mapping rules to normalize category labels.
"""

import pandas as pd
import os

def update_categories():
    """
    Update category names in CSV files according to predefined mapping rules.
    
    Processes data_annotated.csv and data_annotated_with_leaning.csv files,
    applying standardized category naming conventions. Creates backup of 
    original files and reports on changes made.
    
    Category mappings applied:
    - "Social" → "Social Issues"
    - "Corruption/Scandal" → "Corruption & Scandal"
    - "Immigration and Security" → "Immigration & Security"
    
    Raises:
        FileNotFoundError: If required CSV files are not found
        Exception: For CSV reading/writing errors
    """
    csv_files = ["data_annotated.csv", "data_annotated_with_leaning.csv"]
    
    for csv_file in csv_files:
        if not os.path.exists(csv_file):
            print(f"Warning: {csv_file} not found in current directory, skipping...")
            continue
        
        print(f"\nReading {csv_file}...")
        
        try:
            df = pd.read_csv(csv_file)
        except Exception as e:
            print(f"Error reading CSV file {csv_file}: {e}")
            continue
        
        print(f"\nCurrent category distribution in {csv_file}:")
        print(df['Categories'].value_counts())
        
        category_mappings = {
            "Social": "Social Issues",
            "Corruption/Scandal": "Corruption & Scandal", 
            "Immigration and Security": "Immigration & Security"
        }
        
        changes_made = {}
        
        for old_category, new_category in category_mappings.items():
            count = (df['Categories'] == old_category).sum()
            if count > 0:
                df.loc[df['Categories'] == old_category, 'Categories'] = new_category
                changes_made[old_category] = {'new_name': new_category, 'count': count}
                print(f"Updated {count} rows: '{old_category}' → '{new_category}'")
            else:
                print(f"No rows found with category '{old_category}'")
        
        try:
            df.to_csv(csv_file, index=False)
            print(f"\nSuccessfully updated {csv_file}")
        except Exception as e:
            print(f"Error saving CSV file {csv_file}: {e}")
            continue
        
        print(f"\nUpdated category distribution in {csv_file}:")
        print(df['Categories'].value_counts())
        
        if changes_made:
            print(f"\nSummary of changes made to {csv_file}:")
            for old_name, info in changes_made.items():
                print(f"  - {info['count']} rows changed from '{old_name}' to '{info['new_name']}'")
        else:
            print(f"\nNo changes were made to {csv_file} - target categories not found.")
            
        print("\n" + "="*60)

if __name__ == "__main__":
    """
    Main execution block - runs category update process when script is called directly.
    """
    update_categories()
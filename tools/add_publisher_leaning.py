import pandas as pd
import os

publisher_leaning = {
    "CNBC": "Center-Left",
    "Daily News": "Left",
    "POLITICO": "Center-Left",
    "Newsday": "Center-Left",
    "The New York Times": "Left",
    "New York Post": "Right",
    "Business Insider": "Center-Left",
    "Los Angeles Times": "Center-Left",
    "Raw Story": "Left",
    "U.S. News & World Report": "Center-Right",
    "Washington Times": "Right",
    "CBS News": "Center-Left",
    "Fox News": "Right",
    "Bloomberg Business": "Center-Left",
    "HuffPost": "Left",
    "Yahoo! Finance": "Centrist",
    "Breitbart": "Right",
    "TheWrap": "Centrist",
    "Fox Business": "Right",
    "The Boston Globe": "Center-Left",
    "MSNBC.com": "Left",
    "Washington Post": "Center-Left",
    "The New Yorker": "Left",
    "Chicago Tribune": "Center-Right",
    "Forbes": "Center-Right",
    "thedispatch.com": "Center-Right",
    "AP NEWS": "Centrist",
    "Vox": "Left",
    "Los Angeles Daily News": "Center-Left",
    "ABC News": "Center-Left",
    "USA Today": "Centrist",
    "The Epoch Times": "Right",
    "Boston Herald": "Center-Right",
    "Reuters": "Centrist",
    "The Atlantic": "Left"
}

def add_publisher_leaning(input_file, output_file):
    try:
        df = pd.read_csv(input_file)
        
        df['publisher_leaning'] = df['source'].map(publisher_leaning)
        
        unmapped_publishers = df[df['publisher_leaning'].isna()]['source'].unique()
        if len(unmapped_publishers) > 0:
            print("Warning: The following publishers were not found in the leaning dictionary:")
            for publisher in unmapped_publishers:
                print(f"  - {publisher}")
            print("These entries will have 'Unknown' as publisher_leaning")
            
            df['publisher_leaning'] = df['publisher_leaning'].fillna('Unknown')
        
        df.to_csv(output_file, index=False)
        
        print(f"Successfully created {output_file} with publisher leaning information")
        print(f"Total rows processed: {len(df)}")
        
        print("\nPublisher leaning distribution:")
        leaning_counts = df['publisher_leaning'].value_counts()
        for leaning, count in leaning_counts.items():
            print(f"  {leaning}: {count}")
        
        return df
        
    except FileNotFoundError:
        print(f"Error: Could not find the file {input_file}")
        return None
    except Exception as e:
        print(f"Error processing file: {str(e)}")
        return None

if __name__ == "__main__":
    input_file = "data_annotated.csv"
    output_file = "data_annotated_with_leaning.csv"
    
    result_df = add_publisher_leaning(input_file, output_file)
    
    if result_df is not None:
        print(f"\nFirst few rows of the updated dataset:")
        print(result_df[['title', 'source', 'publisher_leaning']].head())
# Usage Guide

## Quick Start

### Prerequisites
- Python 3.8+
- Virtual environment (recommended)

### Setup Instructions

1. **Clone the repository**
```bash
git clone <repository-url>
cd Politician_Media_Coverage
```

2. **Create and activate virtual environment**
```bash
python -m venv venv
source venv/bin/activate  # On Windows: .\venv\Scripts\Activate.ps1
```

3. **Install dependencies**
```bash
pip install pandas numpy scikit-learn matplotlib seaborn
```

4. **Optional: Install spaCy for NER**
```bash
pip install spacy
python -m spacy download en_core_web_sm
```

### Running the Analysis

**Main TF-IDF Analysis**:
```bash
python src/tfidf.py
```

**Data Processing Tools**:
```bash
# Add publisher bias classifications
python tools/add_publisher_leaning.py

# Separate data by category
python tools/separate_by_category.py

# Update category names
python tools/update_categories.py
```

## Expected Output

### Files Generated
- `Visualizations/tfidf_visualization_manual.png` - Main category analysis chart
- `Visualizations/tfidf_political_leaning_manual.png` - Political bias comparison
- `Visualizations/tfidf_political_leaning_manual.txt` - Detailed results

### Console Output
```
Using Manual Entity Normalization
Loaded 500 articles
Dataset shape: (500, 7)
Categories: 8 unique categories

TF-IDF RESULTS BY CATEGORY
Category: Economy
1. tax                       0.1002
2. economy                   0.0692
...
```

## Configuration Options

### TF-IDF Parameters
In `src/tfidf.py`, modify these variables:
```python
USE_NER = False          # True for spaCy NER, False for manual normalization
number_idf_words = 10    # Number of top keywords to extract

# TF-IDF Vectorizer settings
max_features=1000        # Maximum vocabulary size
min_df=2                 # Minimum document frequency
max_df=0.8              # Maximum document frequency
```

### Custom Stop Words
Add domain-specific stop words:
```python
custom_stop_words = set(ENGLISH_STOP_WORDS).union({
    'your', 'custom', 'stop', 'words'
})
```

## Data Format Requirements

Your dataset should be a CSV file with these columns:
- `title` - Article headline
- `body` - Article content
- `source` - Publisher name
- `Categories` - Content category
- `publisher_leaning` - Political bias (optional)

## Troubleshooting

**Common Issues**:

1. **Missing dependencies**: Install all required packages
2. **spaCy model not found**: Download the English model
   ```bash
   python -m spacy download en_core_web_sm
   ```
3. **File not found**: Ensure data files are in the correct directory
4. **Memory issues**: Reduce dataset size or adjust TF-IDF parameters

**Performance Tips**:
- Use manual normalization for faster processing
- Adjust `max_features` parameter for large datasets
- Consider sampling large datasets for initial testing
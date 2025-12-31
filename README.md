# Political Media Coverage Analysis

This repository contains my personal contributions to a group project analyzing the media coverage of California Governor Gavin Newsom across North American news outlets over a one-month period.

## Project Overview

This project implements text mining techniques to analyze political media coverage, featuring TF-IDF analysis with custom entity normalization, political bias detection, and category-based content analysis. The codebase demonstrates proficiency in data preprocessing, natural language processing, and quantitative media analysis.

## Core Components

### Primary Analysis
- **[`src/tfidf_my.py`](src/tfidf_my.py)** - TF-IDF analysis system with dual normalization approaches (manual entity mapping and spaCy NER), custom stop word filtering, and political leaning-based analysis

### Data Processing Utilities
- **[`tools/add_publisher_leaning.py`](tools/add_publisher_leaning.py)** - Maps news publishers to political bias categories using a locally-developed publisher leaning dictionary
- **[`tools/separate_by_category.py`](tools/separate_by_category.py)** - Segments dataset into category-specific CSV files for targeted analysis
- **[`tools/update_categories.py`](tools/update_categories.py)** - Standardizes category naming conventions across datasets

## Implementation

### TF-IDF Analysis System (`src/tfidf_my.py`)


**Text Normalization Strategies:**
- **Entity Mapping Approach**: Manual entity normalization using regex patterns for consistent entity representation
- **NER Approach**: spaCy-powered Named Entity Recognition with entity standardization

**Key Features:**
- Custom stop word filtering (removes temporal terms, common reporting language)
- Entity consolidation (maps variations like "Governor Newsom", "Gavin Newsom", "Gov. Newsom" to single entities)
- Political leaning analysis (Left/Right/Neutral groupings)
- Category-based content analysis
- Configurable TF-IDF parameters with noise reduction

**Analysis Capabilities:**
- Top keyword extraction by category
- Political bias comparison across content
- Document frequency filtering
- Statistical significance validation

## Analysis Flow: Step-by-Step Process

The TF-IDF analysis follows a structured pipeline that transforms raw news articles into meaningful insights:

### 1. **Data Ingestion & Processing**
```python
df = pd.read_csv('data_annotated.csv')
categories = df['Categories'].unique()
```
Load dataset, identify content categories, validate data integrity.

### 2. **Text Preprocessing**
**Entity Normalization** (dual approach):
- **Manual**: Regex-based entity mapping for consistent representation
  ```python
  'newsom_entity': ['newsom', 'gavin newsom', 'governor newsom']
  ```
- **NER**: spaCy entity recognition for PERSON, GPE, ORG standardization

**Text Cleaning**: Remove non-alphabetic characters, normalize case/whitespace, combine title and body.

### 3. **Stop Word Filtering**
Enhanced filtering for news-specific vocabulary:
```python
custom_stop_words = ENGLISH_STOP_WORDS.union({
    'said', 'according', 'report',      # Reporting language
    'week', 'monday', 'january',        # Temporal terms  
    'would', 'could', 'also'           # Common qualifiers
})
```

### 4. **TF-IDF Vectorization**
```python
TfidfVectorizer(
    max_features=1000,    # Reduce noise
    min_df=2, max_df=0.8, # Frequency filtering
    stop_words=custom_stop_words
)
```

### 5. **Analysis Execution**
- **Category Analysis**: Filter by category → preprocess → generate TF-IDF matrix → extract top keywords
- **Political Leaning** (optional): Group by Left/Right/Neutral → separate TF-IDF analysis → compare perspectives

### 6. **Output Processing**
- Clean entity markers and duplicates
- Rank by TF-IDF scores
- Generate structured results with statistical validation

**Example Output:**
```
Category: Economy → Top words: economy (0.234), inflation (0.198), budget (0.156)
Political: Left [climate, healthcare] vs Right [border, taxes]
```
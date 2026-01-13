# Political Media Coverage Analysis

NLP analysis framework for studying political media coverage through TF-IDF analysis and bias detection. This repository contains my contributions to a group project analyzing California Governor Gavin Newsom's media coverage across North American news outlets (Oct-Nov 2025).

## Quick Start

### Installation

```bash
git clone <repository-url>
cd Politician_Media_Coverage
pip install -r requirements.txt
```

### Usage
```bash
# Run TF-IDF analysis
python src/tfidf.py

# Process data with tools
python tools/add_publisher_leaning.py
python tools/separate_by_category.py
python tools/update_categories.py

# Run tests
pytest tests/
```

## Project Overview

My work on the project implements text mining techniques to analyze political media coverage, featuring TF-IDF analysis with custom entity normalization, political bias detection, and category-based content analysis. 

**Key Technologies & Libraries:**
- **Data Processing**: pandas, numpy for dataset manipulation and numerical operations
- **NLP & Text Mining**: scikit-learn (TF-IDF vectorization), spaCy (Named Entity Recognition), regex patterns
- **Visualization**: matplotlib, seaborn for statistical plots and data visualization
- **Analysis Techniques**: TF-IDF statistical modeling, entity normalization, political bias classification


**[Technical Details](docs/TECHNICAL.md)** - Implementation deep-dive and analysis pipeline

**[Data Documentation](docs/DATA.md)** - Dataset structure, collection methodology, and handling

## Core Components
### Primary Analysis
- **[`src/tfidf.py`](src/tfidf.py)** - TF-IDF analysis system with dual normalization approaches (manual entity mapping and spaCy NER), custom stop word filtering, and political leaning-based analysis

### Data Processing Utilities
- **[`tools/add_publisher_leaning.py`](tools/add_publisher_leaning.py)** - Maps news publishers to political bias categories using a locally-developed publisher leaning dictionary
- **[`tools/separate_by_category.py`](tools/separate_by_category.py)** - Segments dataset into category-specific CSV files for targeted analysis
- **[`tools/update_categories.py`](tools/update_categories.py)** - Standardizes category naming conventions across datasets

## Results

**500 articles analyzed** across 8 categories with clear political bias patterns:

![TF-IDF Analysis Results](visualizations/tfidf_visualization_manual.png)

**Top Findings**:
- **Trump** dominates coverage across all political leanings
- **Economic coverage** focuses on taxation and affordability  
- **Climate articles** emphasize policy and energy issues
- **Clear differentiation** between Left/Right keyword priorities

## Testing

The project includes pytest tests for the data processing tools in [`tests/test_tools.py`](tests/test_tools.py). Tests ensure reliable functionality for publisher leaning assignment, category separation, and data updates during refactoring.

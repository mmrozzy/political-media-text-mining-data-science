# Data Documentation

## Data Handling Policy

**Note**: The actual news article dataset is **not included** in this repository. The focus of the repo is on showcasing analytical techniques rather than raw data.

## Dataset Overview

**Collection Period**: October 21 - November 25, 2025  
**Total Articles**: 500 articles analyzing Gavin Newsom coverage  
**Geographic Scope**: North American news outlets  

## Data Structure

**Core Columns**:
- `title` - Article headline text
- `body` - Full article content  
- `source` - Publishing organization
- `Categories` - Content classification
- `publisher_leaning` - Political bias classification

## Category Distribution

| Category | Article Count | Description |
|----------|---------------|-------------|
| National Politics | 185 | Federal-level political coverage |
| Public Image | 77 | Personal and public perception stories |
| Local Politics | 56 | California state and local issues |
| Immigration & Security | 52 | Border, immigration policy coverage |
| Climate & Energy | 40 | Environmental and energy policy |
| Social Issues | 39 | Healthcare, social programs, rights |
| Corruption & Scandal | 33 | Ethics investigations, controversies |
| Economy | 18 | Economic policy, budget, taxation |

## Political Leaning Distribution

| Leaning | Article Count | Percentage |
|---------|---------------|------------|
| Left | 311 | 62.2% |
| Right | 163 | 32.6% |
| Neutral | 26 | 5.2% |

## Data Collection Methodology

Data was collected through web scraping and news aggregation APIs with attention to:
- **Source diversity**: Multiple political perspectives represented
- **Time consistency**: Uniform collection period
- **Content quality**: Full-text articles with sufficient length

## Reproduction Guidelines

**To reproduce this analysis**:
1. Obtain similar news data through:
   - News aggregation APIs (NewsAPI, GDELT)
   - Web scraping
   - Academic news datasets
   - Media monitoring services

2. Ensure data includes:
   - Full article text (title + body)
   - Publisher information
   - Publication dates within target timeframe
   - Content categorization

(Verify licensing and usage rights for any news content used)
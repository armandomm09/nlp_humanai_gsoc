# Social Media Data Analysis for Mental Health Crisis Detection

## Project Overview
This project aims to analyze social media data to detect signs of mental health distress, substance use, or suicidality using natural language processing (NLP) techniques. The analysis involves extracting, preprocessing, and classifying posts from platforms like Twitter or Reddit.

## Tasks and Deliverables

### Task 1: Social Media Data Extraction & Preprocessing
- **Objective**: Extract posts related to mental health distress, substance use, or suicidality using predefined keywords.
- **Deliverable**: Python script to retrieve and store filtered social media posts in a structured CSV/JSON format. Cleaned dataset ready for NLP analysis.

### Task 2: Sentiment & Crisis Risk Classification
- **Objective**: Classify posts into sentiment categories (Positive, Neutral, Negative) and crisis risk levels (High-Risk, Moderate Concern, Low Concern).
- **Tools**: Utilizes VADER or TextBlob for sentiment analysis and TF-IDF or Word Embeddings for crisis term detection.
- **Deliverable**: Script for sentiment and risk classification. Visualization showing distribution of posts by sentiment and risk category.

### Task 3: Crisis Geolocation & Mapping
- **Objective**: Extract location metadata from posts and visualize regional distress patterns.
- **Tools**: Geocoding using geopy, heatmap generation with Folium.
- **Deliverable**: Python script for geocoding posts and generating heatmap of crisis-related discussions. Visualization of regional distress patterns.

## Tools Used
- Python for scripting and data processing.
- NLTK and TextBlob for NLP tasks such as sentiment analysis.
- Pandas for data manipulation and analysis.
- Folium for interactive map visualizations.
- Geopy for geocoding location data.

## Usage
1. Ensure Python and required libraries are installed (`pip install pandas nltk folium geopy scikit-learn`).
2. Run each task script sequentially or as needed.
3. Input/output paths may need customization based on your environment and data sources.



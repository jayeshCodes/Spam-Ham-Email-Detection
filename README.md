# engr-csc522-fall2023-project

## Overview
The project aims to analyse the performance of different algorithms and embeddings for the task of email spam detection.
This project includes two Python scripts for analyzing email data from the Enron dataset. The first script, `enron_by_date.py`, generates a stacked histogram displaying the counts of spam and ham emails over time. The second script, `enron_word_count.py`, creates a pie chart displaying the percentage of spam vs. ham data as well as the total number of each email type within the Enron dataset, and a histogram showcasing the frequency distribution of word counts for both email types. The `ALDA_word2vec.ipynb` and `ALDA_TFIDF.ipynb` notebooks contain the code for training the data and obtaining the precision, recall and accuracy for different algorithms for the respective embeddings as well as exploratory analysis.

## Requirements

- Python 3.x
- pandas
- matplotlib
- seaborn
- nltk

## Installation

1. Clone GitHub Repository
2. Install Dependencies
```
pip install pandas
pip install matplotlib
pip install seaborn
pip install nltk
```

## Usage
1. Ensure you have the Enron dataset downloaded from the link provided in the *CSC522_Final_Report_P27* document (`enron_spam_data.zip`).
2. Unzip the `enron_spam_data.zip` file within your `engr-csc522-fall2023-P27` directory and locate the `enron_spam_data.csv` file.
4. Open a terminal window and run the following commands, respectively:
```
python3 enron_by_date.py
python3 enron_word_count.py
```
5. To run the notebooks, ensure the `enron_spam_data.csv` file is in the appropriate location before execution. Modify the parsing of the csv file in the notebooks accordingly.

## Considerations
The entire CSV file is read during each instance that the Python scripts are run, so the graphs will take several seconds to generate.

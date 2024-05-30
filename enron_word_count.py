import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
import nltk
from nltk.tokenize import word_tokenize

# Read the Enron dataset from the CSV file
df = pd.read_csv('enron_spam_data.csv')
label_counts = df["Spam/Ham"].value_counts()

# Show ham/spam pie chart
fig, ax_pie = plt.subplots()
ax_pie.pie(label_counts, labels=label_counts.index, autopct='%1.1f%%', startangle=90, colors=['blue', 'red'])
ax_pie.set_title("Email Label Distribution")

count_text = f"Ham: {label_counts['ham']} emails\nSpam: {label_counts['spam']} emails"
ax_pie.annotate(count_text, xy=(0, 1), xytext=(10, -10), ha='left', va='top', textcoords='offset points', color='black', fontsize=10, bbox=dict(boxstyle='round,pad=0.3', edgecolor='black', facecolor='white'))
plt.show()

#Preprocessing to remove punctuations and words like re : from subject and message
df['Subject'].fillna('', inplace=True)
df['Message'].fillna('', inplace=True)
df['Subject'] = df['Subject'].str.replace('[^\w\s]', '', regex=True)
df['Message'] = df['Message'].str.replace('[^\w\s]', '', regex=True)

# Combine 'Subject' and 'Message' into a single column
df['text'] = df['Subject'] + ' ' + df['Message']

# Tokenize the 'text' column
df['tokenized_text'] = df['text'].apply(lambda x: word_tokenize(x))

# Calculate word count for tokenized text
df['word_count'] = df['tokenized_text'].apply(len)

# Convert 'Spam/Ham' column to categorical
df['Spam/Ham'] = df['Spam/Ham'].astype('category')

# Calculate mean word count for ham and spam
ham_mean_word_count = df[df['Spam/Ham'] == 'ham']['word_count'].mean()
spam_mean_word_count = df[df['Spam/Ham'] == 'spam']['word_count'].mean()

# Histogram for Word Count and Frequency
plt.figure(figsize=(10, 6))
sns.histplot(df, x='word_count', hue='Spam/Ham', bins=1500, kde=False, palette={'ham': 'red', 'spam': 'blue'})
plt.xlabel('Word Count')
plt.ylabel('Frequency')
plt.title('Distribution of Word Count for Ham and Spam Emails')
plt.legend(title='Spam/Ham', labels=['Ham', 'Spam'])

# Mean Word Count
plt.annotate(f'Ham Mean: {ham_mean_word_count:.2f}', xy=(ham_mean_word_count, 0), xytext=(ham_mean_word_count, 600), arrowprops=dict(facecolor='black', shrink=0.05), fontsize=10, ha='left')
plt.annotate(f'Spam Mean: {spam_mean_word_count:.2f}', xy=(spam_mean_word_count, 0), xytext=(spam_mean_word_count, 800), arrowprops=dict(facecolor='black', shrink=0.05), fontsize=10, ha='left')

plt.xlim(0, 2000)
plt.show()

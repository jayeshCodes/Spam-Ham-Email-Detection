import pandas as pd
import matplotlib.pyplot as plt

# Read the Enron dataset from the CSV file
df = pd.read_csv('enron_spam_data.csv')

# Convert the 'Date' column to a datetime format
df['Date'] = pd.to_datetime(df['Date'])

# Create columns for spam and ham counts
df['Spam_Count'] = df['Spam/Ham'].apply(lambda x: 1 if x == 'spam' else 0)
df['Ham_Count'] = df['Spam/Ham'].apply(lambda x: 1 if x == 'ham' else 0)

# Group data by date and calculate the sum of spam and ham counts for each date
grouped = df.groupby('Date').agg({'Spam_Count': 'sum', 'Ham_Count': 'sum'}).reset_index()

# Create the stacked histogram with 50 bins and a consistent alpha value of 0.5
plt.figure(figsize=(10, 6))
plt.hist([grouped['Date'], grouped['Date']], bins=50, range=(grouped['Date'].min(), grouped['Date'].max()),
         weights=[grouped['Ham_Count'], grouped['Spam_Count']], stacked=True, color=['blue', 'red'],
         alpha=0.4, label=['Ham', 'Spam'])

plt.xlabel('Date')
plt.ylabel('Email Count')
plt.title('Spam and Ham Email Counts by Date')
plt.legend()
plt.xticks(rotation=45)
plt.show()

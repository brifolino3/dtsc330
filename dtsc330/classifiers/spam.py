import pandas as pd
from sklearn.model_selection import train_test_split
import classifier

"""
This will be a classifier of spam emails, using the data set provided
in this week's work
"""

# read in spam data
df = pd.read_csv('data/spam.csv')

# keep necessary columns - numeric
features = df.drop(columns = ['email_id', 'subject', 'email_text', 'sender_email', 'sender_domain','label'])
labels = df['label']

# train test split
train_features, test_features, train_labels, test_labels = train_test_split(features, labels, test_size = 0.333, random_state = 3)

# set classifier
rc = classifier.ReusableClassifier("xgboost")
rc.train(train_features, train_labels)

prediction = rc.predict(test_features)

count_equal = (prediction == test_labels).sum()
print(count_equal / len(test_labels))
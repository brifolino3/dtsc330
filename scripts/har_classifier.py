import sys, os
print("exe:", sys.executable)
print("cwd:", os.getcwd())
print("path[0]:", sys.path[0])
print("sys.path:", sys.path)

import pandas as pd

from dtsc330 import classifier
from dtsc330.readers import har

har_data = har.HAR('data/motion-and-heart-rate-from-a-wrist-worn-wearable-and-labeled-sleep-from-polysomnography-1.0.0', n_people = 1 )
df = har_data.df.copy()


# absolute acceleration
df['acc_x'] = df['acc_x'].abs()
df['acc_y'] = df['acc_y'].abs()
df['acc_z'] = df['acc_z'].abs()

# rolling window
window_size = 50

df['rolling_hr'] = df['hr'].rolling(window = window_size).mean()
df['rolling_acc_x'] = df['acc_x'].rolling(window = window_size).mean()
df['rolling_acc_y'] = df['acc_y'].rolling(window = window_size).mean()
df['rolling_acc_z'] = df['acc_z'].rolling(window = window_size).mean()

# drop rows where rolling values are NaN
df = df.dropna(subset = ['rolling_hr'])


df = df.iloc[::50].copy()


labels = df['is_sleep'].astype(int)

features = df[['rolling_hr', 'rolling_acc_x', 'rolling_acc_y', 'rolling_acc_z']]


rc = classifier.ReusableClassifier(model_type='random_forest')
score = rc.assess(features, labels)

print("Model accuracy:", score)

# lets do it again for 250

# ---------------------------------------------------------
# SECOND TEST: window size = 250
# ---------------------------------------------------------

window_size2 = 250

df2 = har_data.df.copy()

# absolute acceleration
df2['acc_x'] = df2['acc_x'].abs()
df2['acc_y'] = df2['acc_y'].abs()
df2['acc_z'] = df2['acc_z'].abs()

# rolling features with larger window
df2['rolling_hr'] = df2['hr'].rolling(window = window_size2).mean()
df2['rolling_acc_x'] = df2['acc_x'].rolling(window = window_size2).mean()
df2['rolling_acc_y'] = df2['acc_y'].rolling(window = window_size2).mean()
df2['rolling_acc_z'] = df2['acc_z'].rolling(window = window_size2).mean()

# drop NaNs
df2 = df2.dropna(subset = ['rolling_hr'])

# downsample
df2 = df2.iloc[::250].copy()

# labels & features
labels2 = df2['is_sleep'].astype(int)
features2 = df2[['rolling_hr', 'rolling_acc_x', 'rolling_acc_y', 'rolling_acc_z']]

# run assessment
rc2 = classifier.ReusableClassifier(model_type=  'random_forest')
score2 = rc2.assess(features2, labels2)

print("Model accuracy with larger window:", score2)


import pandas as pd
import zipfile
from dtsc330 import classifier
classifier.ReusableClassifier()
# in a script, no name = main
# don't add good notes
# all good

df = pd.read_csv('data/wine+quality.zip', compression = 'zip', delimiter = ';')
print(df)

# train test split
features = df.drop(columns = ['quality'])

rc = classifier.ReusableClassifier()
print(rc.assess(features, test_labels))



"""Create the best possible classifier of sleep from acceleration
 and heart rate
If we have not finished the associated readers by the end of class, 
you will have to complete these readers yourself
The classifier should be written using the technique used in class:
Code should have docstrings
The outermost code you write should be in a script that imports and 
uses reusable_classifier (or a specific version thereof)
The code should also import and use any necessary readers
Any transformations to the raw data should be done in the reader
As before, please add a short description to your README that returns 
the performance of your model and no more than one paragraph on why you 
chose the features you did"""

from dtsc330.readers import har
from dtsc330.readers import classifier
import numpy as np
import sklearn

def main():
    """load in HAR data, buillding the features / labels,
    and assess the classifier's performance"""

    har = har.HAR('data/motion-and-heart-rate-from-a-wrist-worn-wearable-and-labeled-sleep-from-polysomnography-1.0.0')

    # features - derived from rows / timestamps
    features = har.df.drop(columns = ['timestamp', 'is_sleep'])

    # labels "is_sleep"
    labels = har.df['is_sleep']

    # utilize the reusable classifier to evaluate 
    classifier = classifier.ReusableClassifier(model_type = 'logistic_regression')
    performance = classifier.assess(features, labels)
    print(performance)

if __name__ == '__main__':
    main()
This will contain my work for DTSC 330 -
Big Data & Databases during the Spring 2026
semester @ Duquesne University. 

Homework #1
Set up GitHub & the first repository!

Homework #2 (due Feb. 1st)

How do you fill in the missing dates from the grants data?

You can use the last available date or the date of the proceeding
data collected. This could likely be a misrepresentation, though.
Multiple observations taken @ the "same time" may rupture the 
data.

PI_NAMEs contains multiple names. We can only connect individual 
people. Can you make it so that we can get "grantees"?

Utilize pandas function explode()? I will use that function to send the ID number ( assuming the possibility of 
a second one in one element ) to the following line.

The dates for Articles are problematic. Can you fix them?

I am super unfamiliar with this file type which made this more difficult for me to navigate this question.

Homework #3 ( due Feb. 8th )

"""Create the best possible classifier of sleep from acceleration and heart rate
If we have not finished the associated readers by the end of class, you will have to complete these readers yourself
The classifier should be written using the technique used in class:
Code should have docstrings
The outermost code you write should be in a script that imports and uses reusable_classifier (or a specific version thereof)
The code should also import and use any necessary readers
Any transformations to the raw data should be done in the reader
As before, please add a short description to your README that returns the performance of your model and no more than one paragraph on why you chose the features you did"""

This loads wrist-worn heart rate & accelerometer data
& prepares it for the task of sleep classification. To find the rolling means for this set, it first converts the raw data into its absolute values to avoid the effects of direction. The rolling means are then computed over their respective fixed windows to approximate the behavior over time. It is downsized to reduce the redundancy, keeping about one observation per second. The sleep labels are separated from its features for the random forest classifier to learn its patterns to distinguish one's
sleep from wakefulness. 

Homework #4 ( due Feb. 15th )

Add XGBoost to your reusable classifier
For those who did not structure the assessment to be between people (instead using a simple train_test_split), refactor your code to be between people. You can reference my code.
Compare the performance of XGBoost with Random Forest and add the difference (one sentence) to your README.md.
Install fasttext and embed a single word.
Referencing what we've discussed throughout the course up to this point, create an explanation of machine learning.
You can eventually adapt this into a Medium or LinkedIn post to help prepare for a job search
Nothing shows understanding better than teaching
This should be about one page and must include both a diagram and a description. The weight of one vs the other is up to you. The diagram MUST be your own-- it cannot be taken from the internet. Similarly, I would prefer a wrong answer over one created by a large language model. Think of this as preparation for the Performance Review (Midterm) and Job Interview (Final Exam).
Must be in the form of a Markdown (.md) file with an imported image
I would recommend covering the key concepts:
Features
Labels
Classification vs regression
Parameters
Black box vs open box
Different models that we've discussed and how to choose from amongst them
Data vs features
Training data requirements
Anything else that's interesting

The accuracy of XGB took the classifier from 0.9914893617021276 to 0.9942127659574468 accuracy.

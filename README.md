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

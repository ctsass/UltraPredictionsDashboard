# Ultramarathon Predictions

[UltraSignup](https://ultrasignup.com/), 
the leading marketplace and registration platform for
trail and ultra races in the United States, provides a target time
for participants when they register for a race. The platform also 
displays a runner rank for each participant who completes a
race. Both these numbers are generated by simple formulas
based on comparisons of participant times to winning times. 
(See app for further details.)
These target times are not necessarily
to be interpreted as finish time predictions.

I was intersted in what level of accuracy would be possible for
finish time predictions. This is a difficult problem, as ultra and trail 
racing are inherently high variance activites, due to factors such as tough 
terrain, the potential for bad weather, and long distances. 
In partnership with UltraSignup, 
I was provided with a dataset of 1.4+ million results to test my ideas
against the USU target times as a baseline.

After a lot of exploring, prototyping, rejecting, revising, 
and refining, I settled on two prediction methods. 
The first, MED, 
involved making simple modifications to the UltraSignup target time 
formula. It essentially replaces comparisons to minimum times
with comparisons to median times. The second, XGB, involved
engineering features and implementing an XGBoost
regression.

Both methods easily exceed the baseline accuracy of USU target 
times. This app shows the outcomes of the three schemes on
the portion of the dataset I reserved for testing.
(See app for further information about the dataset and the test set.)
**Note**: All time measurements in 
this app are in hours.

One last fun comment. Three of the 
most famous ultramarathons in the United States happened to 
appear in
the test set: Western States, Hardrock 100 Endurance Run, and
Badwater 135. I suggest checking those out first!

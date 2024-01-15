# Ultramarathon Predictions Dashboard

The dashboard is hosted on Streamlit. Click [here](https://ultra-predictions-dashboard.streamlit.app/) to view.

 [UltraSignup](https://ultrasignup.com/), 
the leading marketplace and registration platform for
trail and ultra races in the United States, provides a target time
for participants when they register for a race. The platform also 
displays a runner rank for each participant who completes a
race. Both these numbers are generated by simple formulas
based on comparisons of participant times to winning times.

UltraSignup and I were intersted in what level of accuracy would be possible for
finish time predictions. This is a difficult problem, as ultra and trail 
racing are inherently high variance activites due to factors such as tough 
terrain, the potential for bad weather, and long distances. 
UltraSignup provided me with a set of 1.4+ million results
from their database to develop my models,
 using the USU target times as a baseline.

After a process of exploring, prototyping, rejecting, and revising,
I finalized two new prediction methods. 
The first, MED, 
involves making simple modifications to the UltraSignup target time 
formula. It essentially replaces comparisons to minimum times
with comparisons to median times. The second, XGB, involves
engineering features and implementing an XGBoost tree-based
regression.

This dashboard shows the outcomes of the three schemes on the portion of
 the dataset I reserved for testing.

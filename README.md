# Worksite Safety Risk Classification


Iove Recursion:

The actual project is implemented in project_USETHIS

Start by looking at ovr.py. This file has a number of classes charged with predicting labels from the dataset columns, then we take a normalized
weighted sum of these predictions to create an overall danger magnitude.

visualization.ipynb - displays some ways to visualize our data classifications/cluster w/ time series forecasting and Neural Network to assign danger magnitude to new data.

app.py - a Flask application classifying data based on High Energy Serious Injury or Fatality (HSIF), Low Energy SIF, Potential SIF, 

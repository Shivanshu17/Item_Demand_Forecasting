# Item_Demand_Forecasting
A project on predicting the demand of items for a chain of stores

## Problem Overview
How can time-series data of item sales be used to forecast the requirements of items in various store locations? We are given with the time-series data representing the sales of  50 items at 10 stores for a period of 5 years. I am to use this data to predict the demand of each item in each store for the next three months.

## Client Case Study
Suppose our client is a company like Walmart or Spencer, with multiple stores in a city and multiple items at each store. In order to save costs in transport and effective management of their item supplies, they need a reliable model to predict the demand for a particular item on a given day. It would be an added benefit for them if the model could flag out the anomalies that it wasn’t able to understand as well. This way, they can put domain experts on the problem to identify what might be the reason for those anomalies (maybe a festival or a holiday, etc). This is exactly what our model would deliver. And this would allow our clients to make more informed choices about their approach for item allocation across different stores, and also encourage efficient transportation and storage strategies. 

## Data Used
Data being used can be found here - https://www.kaggle.com/c/demand-forecasting-kernels-only/overview
Its a clean dataset representing the sales of 50 items across 10 stores for a period of 5 years. 

## Tasks
The primary aim of the model is to use the given data to predict the item demand in the 10 stores for a period of 3 months.
Visualizing the confidence in our predictions against Mean Absolute Error, Mean Squared Error, etc.
Detect and flag the anomalies in sales.
Use ETL, SARIMA, Linear Regression, XGBoost, and LSTM to make learn the temporal patterns of our time series, and perform a   relative analysis of all the algorithms.

## Code Structure

### Data Prepration Part
Parse the dataset according to timestamps
Create plots to represent the data from various time chunks.
Define data_split() for test train splits using cross-validation on a rolling basis.

### Data Visualisation Part
Data Visualisation Part
Create a function plot_confidence() to plot the confidence intervals of the model against various evaluation metrics like MAE, MSE, etc.
Create a function plot_anomalies() to plot the data and identify the anomalies. A list of anomalies is to be also returned by this function.

### Exponential Smoothing
Create functions for single, double, and triple exponential smoothing (if seasonality is detected in the data). Plot the results through confidence interval plots.
Create a function etl_forecasting() to act as wrapper functions for making predictions using the three exponential smoothing functions and plot the result on the confidence graph, as described in chapter 7.7 of [2].

### Stationarity
Create a function for visualizing PACF, ACF plots of the time series, it will also show values of the Dickey-Fuller test to detect the presence of unit-roots.
Create a function differencing() to implement differencing for removing trends and seasonality from the data.

### SARIMA
Create a function that first identifies the values of p,d, and q for the model.
Following which the values are initialized in the StatsModel package implementation of the SARIMA model.
The results are visualized and compared against those produced by Exponential Smoothing.

### Feature Generation
Create a function feature_generator() to create the features for the linear regression model.
Create a function to perform target encoding.
Create a function to perform regularised prediction

### ML/DL Models
Implement the linear regression model
Implement the XGBoost regressor model.
Implement the LSTM model with necessary changes in the data.



###


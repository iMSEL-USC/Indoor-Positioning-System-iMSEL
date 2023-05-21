#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Fri Mar 10 12:01:51 2023
the  source code is from the website (https://enjoymachinelearning.com/blog/multivariate-polynomial-regression-python/) and modified by junlin

"""

# make sure to import all of our modules
from timeit import default_timer as timer
import scipy.io
import numpy as np
# sklearn package
from sklearn.preprocessing import PolynomialFeatures
from sklearn.linear_model import LinearRegression
from sklearn.metrics import mean_squared_error
# dataframes
import pandas as pd
# computation
import numpy as np
# visualization
import matplotlib.pyplot as plt


# load data
training_data = scipy.io.loadmat('training_40.mat')
testing_data = scipy.io.loadmat('testing.mat')
inputs = np.float32(training_data["Xc_black_training"])
targets = training_data["Xw_black_training"]
test_inputs = testing_data["Xc_black_testing"]
test_targets = testing_data["Xw_black_testing"]

z = np.polyfit(inputs[:,0], targets[:,2], 2)
print(z)


# seperate out our x and y values
x_values = inputs
y_values = targets[:,0]

# visual
print(x_values[0], y_values[0])

#define our polynomial model, with whatever degree we want
degree=2

# PolynomialFeatures will create a new matrix consisting of all polynomial combinations 
# of the features with a degree less than or equal to the degree we just gave the model (2)
poly_model = PolynomialFeatures(degree=degree)

# transform out polynomial features
poly_x_values = poly_model.fit_transform(x_values)

# should be in the form [1, a, b, a^2, ab, b^2]
print(f'initial values {x_values[0]}\nMapped to {poly_x_values[0]}')


# x_values = test_inputs
# y_values = test_targets[:,0]

# [1, a=5, b=2940, a^2=25, 5*2940=14700, b^2=8643600]

# let's fit the model
poly_model.fit(poly_x_values, y_values)


# we use linear regression as a base!!! ** sometimes misunderstood **
regression_model = LinearRegression()

regression_model.fit(poly_x_values, y_values)

y_pred = regression_model.predict(poly_x_values)


# coefficients and intercepts of the model 
regression_model.coef_
regression_model.intercept_

print(mean_squared_error(y_values, y_pred, squared=False))

# check our accuracy for each degree, the lower the error the better!
# number_degrees = [1,2,3,4,5]
# plt_mean_squared_error = []
# for degree in number_degrees:
#     poly_model = PolynomialFeatures(degree=degree)
  
#     poly_x_values = poly_model.fit_transform(x_values)
#     poly_model.fit(poly_x_values, y_values)
  
#     regression_model = LinearRegression()
#     regression_model.fit(poly_x_values, y_values)
#     y_pred = regression_model.predict(poly_x_values)
  
#     plt_mean_squared_error.append(mean_squared_error(y_values, y_pred, squared=False))
  
# plt.scatter(number_degrees,plt_mean_squared_error, color="green")
# plt.plot(number_degrees,plt_mean_squared_error, color="red")
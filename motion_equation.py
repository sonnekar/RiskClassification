import numpy as np
import matplotlib.pyplot as plt
from scipy.optimize import curve_fit

# Given data points
x_data = np.array([1, 2, 3, 4, 5, 6, 7, 8, 9, 10, 20, 30, 40, 50, 60, 70, 80, 90, 100, 200, 300, 400, 500, 600,700,800,900,1000,2000,3000,4000,5000,6000,7000,8000,9000,10000,20000])
y_data = np.array([200,90,76,66,59,52,49,47,43, 41, 29,24,21,19,17,16,15,14,13,9.2,7.6,6.5,5.9,5.2,4.9,4.8,4.3,4.1,2.9,2.4,2.1,1.9,1.8,1.7,1.6,1.5,1.4,1.1])

# Define the power-law function to fit
def power_law(x, a, b):
    return a * np.power(x, -b)

# Fit the power-law curve to the data
params, covariance = curve_fit(power_law, x_data, y_data, p0=[200, 1])

# Extract the fitted parameters
a, b = params

# Generate y values using the fitted model
y_fit = power_law(x_data, a, b)

# Plot the data and the fitted line
plt.scatter(x_data, y_data, label='Data', color='red')
plt.plot(x_data, y_fit, label=f'Fitted line: y = {a:.2f} * x^(-{b:.2f})', color='blue')
plt.xlabel('x')
plt.ylabel('y')
plt.title('Power-law Fit to Data')
plt.legend()
plt.grid()
plt.show()

# Print the equation of the fitted line
print(f"Fitted line equation: y = {a:.2f} * x^(-{b:.2f})")

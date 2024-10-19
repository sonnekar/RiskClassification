import numpy as np
import matplotlib.pyplot as plt
from scipy.optimize import curve_fit

# Given data points
x_data = np.array([1, 2, 3, 4, 5, 6, 7, 8, 9, 10, 20, 30, 40, 50, 60, 70, 80, 90, 100, 200, 300, 400,500])
y_data = np.array([500, 260, 180, 120, 100, 80, 70, 65, 56, 51, 25, 19, 12, 10, 8.2, 7, 6.3, 5.7, 5, 2.6, 1.7, 1.3,1])

# Define the power-law function to fit
def power_law(x, a, b):
    return a * np.power(x, -b)

# Fit the power-law curve to the data
params, covariance = curve_fit(power_law, x_data, y_data, p0=[500, 1])

# Extract the fitted parameters
a, b = params

# Generate y values using the fitted model
y_fit = power_law(x_data, a, b)

# Plot the data and the fitted line
plt.scatter(x_data, y_data, label='Data', color='red')
plt.plot(x_data, y_fit, label=f'Fitted line: y = {a:.2f} * x^(-{b:.2f})', color='blue')
plt.xlabel('Weight (lbs)')
plt.ylabel('Height (ft)')
plt.title('Gravity Energy Severity Analysis')
plt.legend()
plt.grid()
plt.show()

# Print the equation of the fitted line
print(f"Fitted line equation: y = {a:.2f} * x^(-{b:.2f})")

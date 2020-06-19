import numpy as np
import scipy
import matplotlib.pyplot as plt

# Funtion to optimize
func = lambda theta: np.sin( 1/2 * theta[0] ** 2 - 1/4 * theta[1] ** 2 + 3) * np.cos( 2 * theta[0] + 1 - np.e ** theta[1])

# Generate values to visualize the function
values = 100

X = np.linspace(-2, 2, values)
Y = np.linspace(-2, 2, values)

Z = np.zeros(shape=(values, values))

for ix, x in enumerate(X):
    for iy, y in enumerate(Y):
        Z[iy, ix] = func([x, y])

# Surface altitude
plt.contourf(X, Y, Z, 100)
plt.colorbar()
#plt.show()

# Initial point for the gradient descent algorithm
theta = np.random.rand(2) * 4 - 2

plt.plot(theta[0], theta[1], 'o', c='red')

# Compute partial derivative
h = 0.001
# Learning rate
lr = 0.001
grad = np.zeros(2)
# Iterations of the algorithm
for _ in range(10000):
    for it, th in enumerate(theta):
        _theta = np.copy(theta)
        _theta[it] = _theta[it] + h

        deriv = (func(_theta) - func(theta)) / h

        grad[it] = deriv

    theta = theta - lr * grad

    if (_ % 100 == 0):
        plt.plot(theta[0], theta[1], 'o', c='red')

    print(func(theta))

plt.show()
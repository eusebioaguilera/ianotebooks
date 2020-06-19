import numpy as np
from sklearn.datasets import load_boston
import matplotlib.pyplot as plt

boston = load_boston()
X = np.array(boston['data'][:,5])
Y = np.array(boston['target'])

plt.scatter(X, Y)

# Ponemos el término libre
X = np.array([np.ones(506), X]).T

# Calculamos la regresión lineal
B = np.linalg.inv(X.T @ X) @ X.T @ Y

# Dibujamos el modelo junto a la nube de puntos
plt.plot([4, 9], [B[0] + B[1] * 4, B[0] + B[1] * 9], c='red')
plt.show()

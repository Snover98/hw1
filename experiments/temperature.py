import numpy as np
from matplotlib import pyplot as plt

X = np.array([400, 450, 900, 390, 550])

T = np.linspace(0.01, 5, 100, True)
P = [[], [], [], [], []]
for t in T:
    X_T = (X/np.min(X))**(-1/t)
    for idx, _ in enumerate(X):
        P[idx].append(X_T[idx]/np.sum(X_T))


for i in range(len(X)):
    plt.plot(T, P[i], label=str(X[i]))

plt.xlabel("T")
plt.ylabel("P")
plt.title("Probability as a function of the temperature")
plt.legend()
plt.grid()
plt.show()
exit()

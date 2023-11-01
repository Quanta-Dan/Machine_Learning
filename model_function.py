import numpy as np
import matplotlib.pyplot as plt


def logstic_function(t, jmax, k, t0, j0):
    return jmax/(1+np.exp(-k*(t-t0)))+j0

t = np.linspace(0,100,100)
jmax = 10
k = 0.2
t0 = 30
j0 = 0

y = logstic_function(t, jmax, k, t0,j0)
plt.plot(t,y)
plt.xlabel('time')
plt.ylabel('current')
plt.title('logistic function model')
plt.savefig('function_model.pdf')
plt.show()
from math import exp
import numpy as np
import matplotlib.pyplot as plt
from scipy import integrate

def analytical_posterior(omega, zeta, sum_y2, n):
    return (zeta + omega)**(-n/2.)*exp(-0.5*sum_y2/(zeta + omega))

zeta = 1.
omega_true = 20.
sum_y2 = 226.633
n = 20
omega_min = 0
omega_max = 100

omega = np.linspace(omega_min, omega_max, 100)
post = np.zeros(len(omega))

for i in range(len(omega)):
    post[i] = analytical_posterior(omega[i], zeta, sum_y2, n)

nrm = integrate.quad(analytical_posterior, a=omega_min, b=omega_max, args=(zeta, sum_y2, n))

plt.plot(omega, post/nrm[0])

chain = np.loadtxt('../../../build/var_est_gauss_noise_softplus_float.chain', delimiter=',')

plt.hist(np.log(1+ np.exp(chain[:,21])),histtype='step', range=(0,100), normed=True, bins=20)
plt.show()

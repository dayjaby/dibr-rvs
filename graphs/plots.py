import numpy as np
import matplotlib
matplotlib.rcParams['text.usetex'] = True
matplotlib.rcParams['text.latex.unicode'] = True
matplotlib.rcParams['text.latex.preamble'] = [r'\usepackage{amsmath}', r'\usepackage{amssymb}']
import matplotlib.pyplot as plt
from scipy.special import binom

m = 20.
n = np.arange(m, 100, 1)

v1 = binom(n+m-1,2*m-1)
v2 = binom(n,m)*np.power(m,(n-m))


plt.plot(n, v1, 'b', label=r"$\binom{n+m-1}{2m-1}, m=20$")
plt.plot(n, v2, 'r', label=r"$\binom{n}{m} m^{n-m}, m=20$")

n = np.arange(4, 100, 4)
v3 = binom(n+n/4.-1,2*n/4.-1)
v4 = binom(n,n/4.)*np.power(n/4.,(n-n/4.))

plt.plot(n, v3, 'b.', label=r"$\binom{n+m-1}{2m-1}, m=\frac{n}{4}$")
plt.plot(n, v4, 'r.', label=r"$\binom{n}{m} m^{n-m}, m=\frac{n}{4}$")
plt.xlabel('n')
plt.yscale('log')
#plt.ylabel('Probability')
plt.legend(loc='upper left', shadow=True)
plt.show()
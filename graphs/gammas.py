import numpy as np
import matplotlib
matplotlib.rcParams['text.usetex'] = True
matplotlib.rcParams['text.latex.unicode'] = True
matplotlib.rcParams['text.latex.preamble'] = [r'\usepackage{amsmath}', r'\usepackage{amssymb}']
import matplotlib.pyplot as plt
from scipy.special import binom

m = 10.
n = np.arange(m+1, 30, 1)

v1 = binom(n-2,m-2)*(n-m)
plt.plot(n, v1, 'b', label=r"$\binom{n+m-1}{2m-1}, m=20$")

n = np.arange(4, 30, 2)
m = 0.5*(n+2)
v3 = binom(n-2,m-2)*(n-m)

plt.plot(n, v3, 'r', label=r"$\binom{n+m-1}{2m-1}, m=\frac{n}{4}$")
plt.xlabel('n')
plt.yscale('log')
#plt.ylabel('Probability')
plt.legend(loc='upper left', shadow=True)
plt.show()
import numpy as np
import matplotlib
import colorsys
matplotlib.rcParams['text.usetex'] = True
matplotlib.rcParams['text.latex.unicode'] = True
matplotlib.rcParams['text.latex.preamble'] = [r'\usepackage{amsmath}', r'\usepackage{amssymb}']
import matplotlib.pyplot as plt
from scipy.special import erf

def get_color(color):
    for hue in range(color):
        hue = 1. * hue / color
        col = [int(x) for x in colorsys.hsv_to_rgb(hue, 1.0, 230)]
        yield "#{0:02x}{1:02x}{2:02x}".format(*col)
color = get_color(15)


mu = np.arange(0, 100, 1)
sigmas = np.arange(0,101, 10)
sigmas[0] = 1
for sigma in sigmas:
    v1 = 0.5*(1+erf(-mu/(np.sqrt(6)*sigma)))
    acolor = next(color)
    plt.plot(mu, v1*v1, 'b', color=acolor, label=r"$\sigma = {}$".format(sigma))

plt.xlabel(r'$\mu$')
plt.ylabel(r'$P(B<0) \cdot P(C<0)$')
plt.legend(loc='upper right', shadow=True)

plt.savefig("erf.png")
plt.show()
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


v = np.arange(3,6,0.1)
m = np.power(2,v)+1
n = (m+1)*3
g = n*n-n*v-10*n+v*v+2*v+33-(n*n+10*n+4*n*v-4*v*v-33-20*v)/(m-1)
g2 = 1/6.0*(n-m)*(n-m+1)*(n+2*m-4)
plt.plot(n, g/g2, 'b', label=r"$\frac{|\gamma '|}{|\gamma|}$")
plt.plot(n, 9/n, 'r', label=r'$\frac{9}{n}$')

plt.xlabel(r'$n$')
plt.legend(loc='upper right', shadow=True)

plt.savefig("complexity3.png")
plt.show()
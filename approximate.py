import argparse
import os
import sys
import numpy as np
from collections import OrderedDict
import matplotlib
import colorsys
matplotlib.rcParams['text.usetex'] = True
matplotlib.rcParams['text.latex.unicode'] = True
matplotlib.rcParams['text.latex.preamble'] = [r'\usepackage{amsmath}', r'\usepackage{amssymb}']
import matplotlib.pyplot as plt
from matplotlib.ticker import MaxNLocator

sys.path.append(os.getcwd())

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description='Find the shortest path')
    parser.add_argument('--infile', nargs='?', type=argparse.FileType('r'), default=sys.stdin)
    parser.add_argument('--line', choices=['horizontal'], default='horizontal')
    parser.add_argument('--gammas',nargs='*',default=None)
    parser.add_argument('-m', default='all')
    args = parser.parse_args()

rLp1 = {}
rLp1rR = {}
k = None
    
for line in args.infile.read().splitlines():
    if len(line)>0:
        v = line.split(' ')
        if k is None:
            k = len(v)/2 - 2
        f = v[0]
        i = 1
        for x in xrange(k+1):
            y,x = int(v[i]), int(v[i+1])
            i += 2
        idx = tuple([int(x) for x in v[1:i]])
        yj, j, yrL, rL, yrR, rR = idx
        if f not in rLp1:
            rLp1[f] = {}
            rLp1rR[f] = {}
        #if j == rL + 5:
        if j not in rLp1[f]:
            rLp1[f][j] = []
        if j not in rLp1rR[f]:
            rLp1rR[f][j] = []
        rLp1[f][j].append(float(v[i]))
        rLp1rR[f][j].append(rR)
        
j = 3

def get_color(hue):
    col = [int(x) for x in colorsys.hsv_to_rgb(hue, 1.0, 230)]
    return "#{0:02x}{1:02x}{2:02x}".format(*col)
        
acolor = get_color(40)

for f in ['depthmse','ssim','dsqm','mse']:
    for j in xrange(5,7):
        plt.plot(rLp1rR[f][j], rLp1[f][j], color=get_color(j/10.0), label=r"$r_L={}, j={}$".format(j-1,j))
    plt.xlabel(r'$r_R$')
    plt.ylabel(f)
    plt.legend(loc='upper left', shadow=True)
    plt.show()
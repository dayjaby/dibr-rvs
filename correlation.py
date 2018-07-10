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
from scipy import stats

sys.path.append(os.getcwd())

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description='Find the shortest path')
    parser.add_argument('--infile', nargs='?', type=argparse.FileType('r'), default=sys.stdin)
    parser.add_argument('--line', choices=['horizontal'], default='horizontal')
    parser.add_argument('--gammas',nargs='*',default=None)
    parser.add_argument('--gammas_exclude',nargs='*',default=[])
    parser.add_argument('-y', type=int, default=0)
    parser.add_argument('-m', default='all')
    args = parser.parse_args()

gamma = {}
k = None
    
horizontal_line = set()
gamma_names = []
for line in args.infile.read().splitlines():
    if len(line)>0:
        v = line.split(' ')
        if k is None:
            k = len(v)/2 - 2
        f = v[0]
        i = 1
        for x in xrange(k+1):
            y,x = int(v[i]), int(v[i+1])
            if y == args.y:
                horizontal_line.add((y,x))
            i += 2
        idx = tuple([int(x) for x in v[1:i]])
        if idx not in gamma: 
            gamma[idx] = {}
        if f not in args.gammas_exclude:
            gamma[idx][f] = float(v[i])
            if f not in gamma_names:
                gamma_names.append(f)
    
values = {g: [] for g in gamma_names}
colors = {g: [] for g in gamma_names}
dists = {}
distAvgs = {}
for idx,v in gamma.items():
    _, j, _, rL, _, rR = idx
    #hue = (np.abs(j-rL)+np.abs(j-rR)-2)/30.0
    hue = 1.0-np.power(1.0/(np.abs(j-rL)+1)+1.0/(np.abs(j-rR)+1),1)

    clr = matplotlib.colors.hsv_to_rgb([hue, 1.0, 1.0])
    if hue not in dists:
        dists[hue] = {f: [] for f in gamma_names}
        distAvgs[hue] = {g: 0.0 for g in gamma_names}
    if len(v)==5:
        for f, value in v.items():
            values[f].append(value)
            colors[f].append(clr)
            dists[hue][f].append(value)

for h in dists.keys():
    for g in gamma_names:
        distAvgs[h][g] = np.mean(dists[h][g])
        
distAvgArr = {g: [distAvgs[h][g]  for h in dists.keys()] for g in gamma_names}
            
#for f1,f2 in [('depthmse','ssim'),('ssim','dsqm'),('mse','ssim'),('mse','dsqm'),('depthmse','mse'),('depthmse','dsqm')]:
for f1,f2,l1,l2 in [('mse','mseapprox','MSE','MSE approximation'),('ssim','ssimapprox','SSIM','SSIM approximation')]:
    slope, intercept, r_value, p_value, std_err = stats.linregress(values[f1], values[f2])
    plt.scatter(values[f1], values[f2], color=colors[f1], s=2)
    print(f1,f2,np.corrcoef(values[f1],values[f2])[0,1])
    plt.xlabel(l1)
    plt.ylabel(l2)
    plt.show()
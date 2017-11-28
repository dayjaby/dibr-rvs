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

import sys
sys.path.append(os.getcwd())
import dibr_single

parser = argparse.ArgumentParser(description='Find a solution for a camera rig')
parser.add_argument('--dir')
parser.add_argument('--out')
parser.add_argument('--infile', nargs='?', type=argparse.FileType('r'), default=sys.stdin)
parser.add_argument('--vinfile', nargs='?', type=argparse.FileType('r'))
parser.add_argument('--line', choices=['horizontal'], default='horizontal')
parser.add_argument('--gammas',nargs='*',default=None)
parser.add_argument('--gammas_exclude',nargs='*',default=None)
parser.add_argument('-y', type=int, default=0)
parser.add_argument('--ys', type=int, default=1)
parser.add_argument('--ym', type=int, default=3)
parser.add_argument('-k', type=int, default=2)
parser.add_argument('-m', default='all')
args = parser.parse_args()

gamma = {}
k = None
    
horizontal_line = set()
    
for line in args.infile.read().splitlines():
    if len(line)>0:
        v = line.split(' ')
        k = len(v)/2 - 2
        f = v[0]
        i = 1
        for x in xrange(k+1):
            y,x = int(v[i]), int(v[i+1])
            if y == args.y:
                horizontal_line.add((y,x))
            i += 2
        idx = tuple([int(x) for x in v[1:i]])
        if f not in gamma: 
            gamma[f] = {}
        gamma[f][idx] = float(v[i])

for line in args.vinfile.read().splitlines():
    if len(line)>0:
        v = line.split(' ')
        k = len(v)/2 - 2
        f = v[0]
        i = 1
        for x in xrange(k+1):
            v[i+1], v[i] = int(v[i]), int(v[i+1])
            i += 2
        idx = tuple([int(x) for x in v[1:i]])
        if f not in gamma: 
            gamma[f] = {}
        gamma[f][idx] = float(v[i])

        
if args.gammas_exclude is not None:
    for f in args.gammas_exclude:
        del gamma[f]
if args.gammas is None:
    gamma_names = gamma.keys()
else:
    gamma_names = args.gammas
#for f in gamma_names:
#    print(gamma[f][(0, 1, 0, 0, 0, 2)])
            
horizontal_line = sorted(horizontal_line)

import networkx as nx

if args.m != 'all':
    ms = [int(args.m)]
graphs = {}
vstarts = {}
vends = {}
ygraphs = {}
yvstarts = {}
yvends = {}
k = args.k
if k==2 and args.line=='horizontal':
    n = len(horizontal_line)
    if args.m == 'all':
        ms = xrange(n/5,n/2)
    for m in ms:
        G=nx.DiGraph()
        vstart = (1,1)
        vend = (m,n)
        G.add_node(vstart)
        G.add_node(vend)
        jrb = [1]
        for i in xrange(2,m+1):
            if i == m:
                jr = [n]
            else:
                jr = xrange(i,n-m+i+1)
            for j in jr:
                G.add_node((i,j))
                for j2 in jrb:
                    if j2<j:
                        weight = {}
                        for f in gamma_names:
                            weight[f] = 0
                            for y in xrange(args.y,args.y+args.ys):
                                weight["{}{}".format(f,y)] = 0.0
                            for v in xrange(j2+1,j):
                                for y in xrange(args.y,args.y+args.ys):
                                    weight["{}{}".format(f,y)] += gamma[f][(y,v-1,y,j2-1,y,j-1)]
                                    weight[f] += gamma[f][(y,v-1,y,j2-1,y,j-1)]
                        G.add_edge((i-1,j2),(i,j),**weight)
                        #print(((i-1,j2),(i,j),weight))
            jrb = jr
        graphs[m] = G
        vstarts[m] = vstart
        vends[m] = vend
    """for rx in xrange(n):
        G=nx.DiGraph()
        vstart = (1,1)
        vend = (args.ym,args.ys)
        G.add_node(vstart)
        G.add_node(vend)
        jrb = [1]
        for i in xrange(2,args.ym+1):
            if i == args.ym:
                jr = [args.ys]
            else:
                jr = xrange(i,args.ys-args.ym+i+1)
            for j in jr:
                G.add_node((i,j))
                for j2 in jrb:
                    if j2<j:
                        weight = {}
                        for f in gamma_names:
                            weight[f] = 0
                            for v in xrange(j2+1,j):
                                weight[f] += gamma[f][(v-1,rx,j2-1,rx,j-1,rx)]
                        G.add_edge((i-1,j2),(i,j),**weight)
                        #print(((i-1,j2),(i,j),weight))
            jrb = jr
        ygraphs[rx] = G
        yvstarts[rx] = vstart
        yvends[rx] = vend"""
    y = args.y

def get_color(color):
    for hue in range(color):
        hue = 1. * hue / color
        col = [int(x) for x in colorsys.hsv_to_rgb(hue, 1.0, 230)]
        yield "#{0:02x}{1:02x}{2:02x}".format(*col)
#approxs = [('ssim',['mse','depthssim','depthmse']), ('mse',['ssim','depthssim','depthmse'])]
approxs = []
color = get_color(len(gamma_names)+4)
ms = np.array(ms,dtype=np.int)
approximations = {}

for f in gamma_names:
    equid = []
    best = [0.12096856682499252, 0.1068054787238439, 0.091188306574026726, 0.082520382956663793, 0.076973542771736778, 0.072147304457426079, 0.068967103487253184, 0.065668687031666451, 0.062239045359690992]
    best2 = []
    iterative = []
    mcolor = next(color)

    
    # graph-based solution
    """for m in ms:
        G = graphs[m]
        p = nx.dijkstra_path(G,vstarts[m],vends[m],weight=f)
        total_weight = 0.0
        p = np.array([v[1]-1 for v in p])
        py = {rx: np.array([v[1]-1 for v in nx.dijkstra_path(ygraphs[rx],yvstarts[rx],yvends[rx],weight=f)]) for rx in p}
        for x in xrange(n):
            rxl = p[p<x]
            if len(rxl) == 0:
                rxl = x
            else:
                rxl = rxl[-1]
            rxr = p[p>x]
            if len(rxr) == 0:
                rxr = x
            else:
                rxr = rxr[0]            
            for y in xrange(args.y,args.y+args.ys):
                axs = [x]
                ays = [y]

                for rx in [rxl,rxr]:
                    if y in py[rx]:
                        axs.append(rx)
                        ays.append(y)
                    else:
                        ryu = py[rx][py[rx]<y]
                        ryd = py[rx][py[rx]>y]
                        axs.append(rx)
                        if len(ryu)>0:
                            ays.append(ryu[-1])
                        else:
                            ays.append(y)
                        axs.append(rx)
                        if len(ryd)>0:
                            ays.append(ryd[0])
                        else:
                            ays.append(y)
                selfR = False
                for rx, ry in zip(axs[1:],ays[1:]):
                    if rx == x and ry == y:
                        selfR = True
                if not selfR:
                    #print(axs,ays)
                    weight = dibr_single.render(args.dir,args.out,str(zip(axs,ays)),axs,ays,f)
                    print("{}@{}:{}".format(f,str(zip(axs,ays)),weight))
                    total_weight += weight
            #total_weight = nx.dijkstra_path_length(G,yvstarts[rx],yvends[rx],weight=f)
        best.append(total_weight)"""

    #plt.plot(ms, equid, '.', color=mcolor, label="{}-equidistant".format(f))
    #print(best)
    color = get_color(2)
    mcolor = next(color)
    plt.plot(ms, np.array(best) / (args.ys * n - args.ym*m) * args.ym * m, color=mcolor, label=r"$g_2$")
    acc = np.array([0.0 for m in ms])
    for y in xrange(args.y,args.y+args.ys):
        b = []
        for m in ms:
            G = graphs[m]
            p = nx.dijkstra_path(G,vstarts[m],vends[m],weight="{}{}".format(f,y))
            total_weight = nx.dijkstra_path_length(G,vstarts[m],vends[m],weight="{}{}".format(f,y)) / ((n-m) * args.ys) * args.ys * m
            p = [v[1] for v in p]
            print("{}            : {}".format(f.ljust(5),p).ljust(60,' ') + " -> {:.7f}".format(total_weight))
            b.append(total_weight)
        acc += np.array(b)
    mcolor = next(color)
    plt.plot(ms, acc, color=mcolor, label=r'$g_1$')

    #plt.plot(ms, best2, '--', color=mcolor, label="{} approximation".format(f))
    plt.gca().xaxis.set_major_locator(MaxNLocator(integer=True))
    plt.xlabel(r'$m_x$')
    #plt.title(f)
    #plt.ylim([min(equid+best),max(equid+best)])
    plt.legend(loc='upper right', shadow=True)
    plt.savefig('2dresult.png', format='png', dpi=900) 
    plt.show()
    # This does, too    ylim = max(equid+best)
    
    for f2, approx_by_arr in approxs:
        if f==f2:
            color2 = get_color(len(approx_by_arr)+1)
            next(color2)
            for approx_by in approx_by_arr:
                ylim = max(ylim,max(approximations[(f,approx_by)]))
                acolor = next(color2)
                plt.plot(ms, approximations[(f,approx_by)], '--', color=acolor, label="{}".format(approx_by))
            
            plt.plot(ms, equid, '.', color='r', label="{}-equidistant".format(f))
            plt.plot(ms, best, color='r', label=f)
            plt.gca().xaxis.set_major_locator(MaxNLocator(integer=True))
            plt.xlabel(r'$m$')
            plt.legend(loc='upper center', shadow=True)
            #plt.title(f)
            plt.show()
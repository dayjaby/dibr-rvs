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
    parser.add_argument('--gammas_exclude',nargs='*',default=None)
    parser.add_argument('-y', type=int, default=0)
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
                            weight["{}v2".format(f)] = 0
                            for v in xrange(j2+1,j):
                                g = horizontal_line[v-1]+horizontal_line[j2-1]+horizontal_line[j-1]
                                gv1 = horizontal_line[v-1]+horizontal_line[j2-1]
                                gv2 = horizontal_line[v-1]+horizontal_line[j-1]
                                weight[f] += gamma[f][g]
                                l = 1.0
                                for rL in xrange(j2-1,v-1):
                                    l*= 1-gamma[f][horizontal_line[rL+1]+horizontal_line[rL]]
                                #weight["{}v2".format(f)] += 1-l
                                l = 1.0
                                for rR in xrange(v-1,j-1):
                                    l*= 1- gamma[f][horizontal_line[rR]+horizontal_line[rR+1]]
                                #weight["{}v2".format(f)] += 1-l
                        G.add_edge((i-1,j2),(i,j),**weight)
                        #print(((i-1,j2),(i,j),weight))
            jrb = jr
        graphs[m] = G
        vstarts[m] = vstart
        vends[m] = vend
    y = args.y

def get_color(color):
    for hue in range(color):
        hue = 1. * hue / color
        col = [int(x) for x in colorsys.hsv_to_rgb(hue, 1.0, 230)]
        yield "#{0:02x}{1:02x}{2:02x}".format(*col)
#approxs = [('ssim',['mse','depthssim','depthmse']), ('mse',['ssim','depthssim','depthmse'])]
approxs = []
color = get_color(len(gamma_names)+25)
ms = np.array(ms,dtype=np.int)
approximations = {}

for f in gamma_names:
    equid = []
    best = []
    best2 = []
    iterative = []
    mcolor = next(color)
    
    # iterative solution
    """if k==2 and args.line=='horizontal':
        for m in ms:
            R = np.array([1,n])
            for i in xrange(2,m):
                opt_r = None
                opt_costs = None
                for r in xrange(2,n):
                    costs = 0.0
                    if r not in R:
                        R2 = np.append(R,r)
                        R2 = np.sort(R2)
                        rL = R2[0]
                        for rR in R2[1:]:
                            for j in xrange(rL+1,rR):
                                costs += gamma[f][(y,j-1,y,rL-1,y,rR-1)]
                            rL = rR
                        if opt_r is None or costs < opt_costs:
                            opt_r = r
                            opt_costs = costs
                R = np.append(R,opt_r)     
                R = np.sort(R)
            total_costs = 0.0
            rL = R[0]
            for rR in R[1:]:
                for j in xrange(rL+1,rR):
                    total_costs += gamma[f][(y,j-1,y,rL-1,y,rR-1)]
                    #print("{}:{}".format((j-1,rL-1,rR-1),gamma[f][(y,j-1,y,rL-1,y,rR-1)]))
                rL = rR
            total_costs /= (n-m)
            iterative.append(total_costs)
            print("{}-iterative: {}".format(f.ljust(5),list(R)).ljust(60,' ') + " -> {:.7f}".format(total_costs))
        plt.plot(ms, iterative, "--", color=mcolor, label="{}-iterative".format(f), linewidth=2.5)"""
          
    # improving iterative solution
    """if k==2 and args.line=='horizontal':
        for m in ms:
            R = list(np.linspace(1,n,m,dtype=np.int32))
            for i in xrange(2,m):
                opt_r = None
                opt_i = None
                opt_rn = None
                opt_r_prime = None
                opt_costs = 0.0
                for i, r in list(enumerate(R))[1:-1]:
                    costs = 0.0
                    rL = R[i-1]
                    rR = R[i+1]
                    for j in xrange(rL+1,r):
                        costs -= gamma[f][(y,j-1,y,rL-1,y,r-1)]
                    for j in xrange(r+1,rR):
                        costs -= gamma[f][(y,j-1,y,r-1,y,rR-1)]
                    costs2 = 0.0
                    opt_rn2 = None
                    opt_costs2 = None
                    for rn in xrange(rL+1,rR):
                        if rn != r:
                            for j in xrange(rL+1,rn):
                                costs2 += gamma[f][(y,j-1,y,rL-1,y,rn-1)]
                            for j in xrange(rn+1,rR):
                                costs2 += gamma[f][(y,j-1,y,rn-1,y,rR-1)]
                            if opt_rn2 is None or costs2 < opt_costs2:
                                opt_rn2 = rn
                                opt_costs2 = costs2
                    if opt_costs2 is not None:
                        costs+=opt_costs2
                        if costs < opt_costs:
                            print("{} better than {}".format(opt_rn2,r))
                            opt_r = r
                            opt_i = i
                            opt_rn = opt_rn2
                            opt_costs = costs
                if opt_r is not None:
                    R[opt_i] = opt_rn
            total_costs = 0.0
            rL = R[0]
            for rR in R[1:]:
                for j in xrange(rL+1,rR):
                    total_costs += gamma[f][(y,j-1,y,rL-1,y,rR-1)]
                    #print("{}:{}".format((j-1,rL-1,rR-1),gamma[f][(y,j-1,y,rL-1,y,rR-1)]))
                rL = rR
            total_costs /= (n-m)
            iterative.append(total_costs)
            print("{}-iterative: {}".format(f.ljust(5),list(R)).ljust(60,' ') + " -> {:.7f}".format(total_costs))
        plt.plot(ms, iterative, "--", color=mcolor, label="{}-iterative".format(f), linewidth=2.5)"""
        
    # equidistant solution
    for m in ms:
        G = graphs[m]
        weight = nx.get_edge_attributes(G,f)
        total_weight = 0
        p = list(np.linspace(1,n,m,dtype=np.int32))
        last_node = (1,1)
        for i in p[1:]:
            next_node = (last_node[0]+1,i)
            total_weight += weight[(last_node,next_node)]
            last_node = next_node
        total_weight /= (n-m)
        equid.append(total_weight)
        print("{}-equidistant: {}".format(f.ljust(5),p).ljust(60,' ') + " -> {:.7f}".format(total_weight))
        
    # graph-based solution
    for m in ms:
        G = graphs[m]
        p = nx.dijkstra_path(G,vstarts[m],vends[m],weight=f)
        total_weight = nx.dijkstra_path_length(G,vstarts[m],vends[m],weight=f) / (n-m)
        best.append(total_weight)
        p = [v[1] for v in p]
        print("{}            : {}".format(f.ljust(5),p).ljust(60,' ') + " -> {:.7f}".format(total_weight))
        weight = nx.get_edge_attributes(G,f)
        last_node = (1,1)
        for i in p[1:]:
            next_node = (last_node[0]+1,i)
            #for j in xrange(last_node[1]+1,next_node[1]):
            #    print("{}:{}".format((j-1,last_node[1]-1,next_node[1]-1),gamma[f][(y,j-1,y,last_node[1]-1,y,next_node[1]-1)]))
            last_node = next_node
            
    # graph-based solution (approximation)
    for m in ms:
        G = graphs[m]
        p = nx.dijkstra_path(G,vstarts[m],vends[m],weight="{}v2".format(f))
        total_weight = nx.dijkstra_path_length(G,vstarts[m],vends[m],weight="{}v2".format(f)) / (n-m)
        p = [v[1] for v in p]
        weight = nx.get_edge_attributes(G,f)
        last_node = (1,1)
        total_weight = 0
        for i in p[1:]:
            next_node = (last_node[0]+1,i)
            total_weight += weight[(last_node,next_node)]
            last_node = next_node            
        total_weight /= (n-m)
        print("{} approximat : {}".format(f.ljust(5),p).ljust(60,' ') + " -> {:.7f}".format(total_weight))
        best2.append(total_weight)

    for f2, approx_by_arr in approxs:
        if f2==f:
            for approx_by in approx_by_arr:
                approximations[(f,approx_by)] = []
    for m in ms:
        for f2, approx_by_arr in approxs:
            if f2==f:
                for approx_by in approx_by_arr:
                    G = graphs[m]
                    p = nx.dijkstra_path(G,vstarts[m],vends[m],weight=approx_by)
                    p = [v[1] for v in p]
                    weight = nx.get_edge_attributes(G,f)
                    total_weight = 0
                    last_node = (1,1)
                    for i in p[1:]:
                        next_node = (last_node[0]+1,i)
                        total_weight += weight[(last_node,next_node)]
                        last_node = next_node
                    total_weight /= (n-m)
                    approximations[(f,approx_by)].append(total_weight)
                    print("{}->{}    : {}".format(approx_by,f.ljust(5),p).ljust(60,' ') + " -> {:.7f}".format(total_weight))
    plt.plot(ms, equid, '.', color=mcolor, label="{}-equidistant".format(f))
    plt.plot(ms, best, color=mcolor, label=f)
    #plt.plot(ms, best2, '--', color=mcolor, label="{} approximation".format(f))
    plt.gca().xaxis.set_major_locator(MaxNLocator(integer=True))
    plt.xlabel(r'$m$')
    #plt.title(f)
    #plt.ylim([min(equid+best),max(equid+best)])
    plt.legend(loc='upper center', shadow=True)
    plt.show()
    ylim = max(equid+best)
    
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
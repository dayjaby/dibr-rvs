import numpy as np
import matplotlib.pyplot as plt
import networkx as nx
import random

def is_strict_sorted(lst, key=lambda x: x):
    for i, el in enumerate(lst[1:]):
        if key(el) <= key(lst[i]): # i is the index of the previous element
            return False
    return True

def dijkstra(G,root,target,weight_fn):
    dist = {root:0}
    prev = {}
    Q = list(G.nodes())
    weight_cache = {}
    while len(Q)>0:
        min_dist = None
        u = None
        for v in Q:
            if v in dist and (min_dist is None or dist[v]<min_dist):
                min_dist = dist[v]
                u = v
        if u==target:
            break
        Q.remove(u)
        for v in G.successors(u):
            h, w = weight_fn(u,v)
            if h not in weight_cache:
                weight_cache[h] = w(h)
            new_dist = dist[u] + weight_cache[h]
            if v not in dist or new_dist < dist[v]:
                dist[v] = new_dist
                prev[v] = u
    v = np.array(weight_cache.values())
    print("Avg weight: {}, Std weight: {}".format(np.mean(v),np.std(v)))
    return dist, prev, weight_cache

def rvs(n,m,k=2):
    all_rvs = np.indices((n,)*m).reshape(m,-1).T # the set {0,...,n-1}^m
    valid_rvs = [rvs for rvs in all_rvs if is_strict_sorted(rvs)]

    G=nx.DiGraph()
    root = ('root')
    count_y = [0]*n
    layer = {root:0}
    pos = {root:(0,0)}
    all_calculations = set()
    for i,rvs in enumerate(valid_rvs):
        rvs = tuple(rvs)
        node = ('rvs',) + rvs
        layer[node] = 1
        G.add_edge(root, node)
        calculations = []
        pos[node] = (1,i)
        for x in set(range(n)) - set(rvs):
            if k == 2:
                if x < rvs[0]: # extrapolation left
                    calculations.append((x,rvs[0],rvs[1]))
                elif x > rvs[-1]: # extrapolation right
                    calculations.append((x,rvs[-2],rvs[-1]))
                else: # interpolation
                    for i in range(len(rvs)):
                        if rvs[i]<x and rvs[i+1]>x:
                            calculations.append((x,rvs[i],rvs[i+1]))
                            break
            elif k == 1:
                for r in rvs:
                    calculations.append((x,r,-1))
        G[root][node]['calcs'] = set(calculations)
        all_calculations |= G[root][node]['calcs']

    allC = set()
    for y in range(n-1):
        for z in range(y+1,n):
            for x in range(n):
                if x != y and x != z:
                    allC.add((x,y,z))

    #print(allC - all_calculations)
    i=0
    while True:
        max_children = []
        max_node = None
        max_gamma = None
        for node in G.nodes():
            children = G.successors(node)
            count = {}
            for child in children:
                for calc in G[node][child]['calcs']:
                    if calc in count:
                        count[calc].append(child)
                    else:
                        count[calc] = [child]
            nmax_children = []
            nmax_gamma = None
            for key, value in count.items():
                if len(value)>len(nmax_children):
                    nmax_children = value
                    nmax_gamma = key
            if len(nmax_children)>len(max_children):
                max_node = node
                max_gamma = nmax_gamma
                max_children = nmax_children
                i=i+1
        if len(max_children) == 1:
            print("{} iterations needed".format(i))
            break
        calc_node = ('calc',max_node,max_gamma)
        G.add_edge(max_node,calc_node)
        G[max_node][calc_node]['calcs'] = set([max_gamma])
        layer[calc_node] = layer[max_node] + 1
        pos[calc_node] = (layer[calc_node], pos[max_children[0]][1])
        for node in max_children:
            #G.remove_node(max_node,node)
            G.add_edge(calc_node,node)
            G[calc_node][node]['calcs'] = G[max_node][node]['calcs'] - set([max_gamma])
            layer[node] = layer[calc_node] + 1
            G.remove_edge(max_node,node)
            pos[node] = (layer[calc_node]+1, pos[node][1])

    for u,v,data in G.edges(data=True):
        if len(data['calcs']) > 1 and layer[u] is not None:
            s = sorted(data['calcs'].copy(), key=lambda calc:-abs(calc[0]-calc[1])-abs(calc[0]-calc[2]))
            x = u
            while len(s)>1:
                gamma = s.pop()
                calc_node = ('calc',x,gamma)
                G.add_edge(x,calc_node)
                G[x][calc_node]['calcs'] = set([gamma])
                layer[calc_node] = layer[x] + 1
                pos[calc_node] = (layer[calc_node], pos[x][1])
                x = calc_node
            G.remove_node(v)
            if len(s)>0:
                gamma = s.pop()
                G.add_edge(x,v)
                G[x][v]['calcs'] = set([gamma])
                layer[v] = layer[x] + 1
                pos[v] = (layer[v], pos[x][1])

    count_children = {}
    for l in range(0,n-m):
        for node in G.nodes():
            if layer[node] == l:
                children = G.successors(node)
                parents = G.predecessors(node)
                if len(parents)>0:
                    count_children[node] = len(children) * count_children[parents[0]]
                else:
                    count_children[node] = len(children)
                for i,child in enumerate(children):
                    pos[child] = (pos[node][0]+1,pos[node][1]+float(i)/count_children[node])

    target = ('target')
    layer[target] = 1
    for node in G.nodes():
        if len(G.successors(node)) == 0:
            G.add_edge(node,target)
            G[node][target]['calcs'] = set()
            G[node][target]['weight'] = 0
            layer[target] = max(layer[target], layer[node]+1)
    pos[target] = (layer[target], pos[root][1])

    top_calculations = set()
    for u,v,data in G.edges([root],data=True):
        top_calculations |= data['calcs']

    def rvs_dijkstra(iterations=1,weight_fn=None,std=1):
        if weight_fn is None:
            #weight_fn = lambda x,y,z: abs(np.random.normal(0,std*0.2*(abs(x-y)+abs(x-z))))
            weight_fn = lambda x,y,z: abs(abs(x-y)+abs(x-z))
            #weight_fn = lambda x,y,z: abs(np.random.normal(0,std))
        def edge_weight(u,v):
            if v==target:
                h = u
                w = lambda h: 0
            else:
                h = x,y,z = list(G[u][v]['calcs'])[0]
                w = lambda h: weight_fn(*h)
            return h,w

        gammas = []
        path = []
        for i in range(iterations):
            del path[:]
            d,p,c = dijkstra(G,root,target,edge_weight)
            x = target
            while x!=root:
                x = p[x]
                path.append(x)
            path = path[::-1]
            gammas.append(len(c))
        return path[-1], np.mean(gammas)
    return rvs_dijkstra, len(all_calculations), len(top_calculations)+n-m-1

def test_optimal_tree():
    for m in range(3,6):
        ns = range(m+1,15)
        avgs = []
        worsts = []
        bests = []
        for n in ns:
            avg, worst, best = rvs(n,m,iterations=50)
            avgs.append(avg/worst)
            worsts.append(1)
            bests.append(best/worst)
        plt.plot(ns, avgs, label="average", color=(1.0-(m-3)*0.2,0,0))
        plt.plot(ns, bests, label="best", color=(0,0,1.0-(m-3)*0.2))
    plt.xlabel(r'number of cameras in a row (n)')
    plt.ylabel(r'$\gamma $s needed compared to worst case')
    plt.show()

def test_draw_graph():
    if False==True:
        plt.figure(1,figsize=(20,12))

        nx.draw_networkx_nodes(G, pos, node_size=20)
        """nl = []
        pnl = []
        for u,v,data in G.edges(data=True):
            if pos[u][0] == pos[v][0]:
                nl.append(u)
                nl.append(v)
                pnl.append(parent[u])
                pnl.append(parent[v])
                print(u,v)
                print(parent[u],parent[v])
        nx.draw_networkx_nodes(G, pos, nodelist=nl, node_color='b', node_size=20)
        nx.draw_networkx_nodes(G, pos, nodelist=pnl, node_color='y', node_size=20)"""
        nx.draw_networkx_edges(G, pos, width=1)

        plt.axis('off')
        plt.savefig("weighted_graph.png") # save as png
        plt.show() # display

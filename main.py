#!/usr/bin/env python
"""
An example using Graph as a weighted network.
"""
# Author: Aric Hagberg (hagberg@lanl.gov)
try:
    import matplotlib.pyplot as plt
except:
    raise

import networkx as nx
import random

gridWidth = 10
gridHeight = 10


def getNeighbors(v):
    x,y = v
    n = []
    if x>0:
        n.append((x-1,y))
    if x<gridWidth-1:
        n.append((x+1,y))
    if y>0:
        n.append((x,y-1))
    if y<gridHeight-1:
        n.append((x,y+1))
    return n


G=nx.Graph()
for x in range(gridWidth):
    for y in range(gridHeight):
        v = (x,y)
        for n in getNeighbors(v):
            G.add_edge(v, n, weight=random.random())

sortedEdges = {(u,v):d for (u,v,d) in sorted(G.edges(data=True),key=lambda d:d[2]['weight'])}
consumedEdges = []
consumedNodes = {}
currentTree = 0
trees = {}
pos={v:v for v in G.nodes()}
for e,d in sortedEdges.items():
    u,v=e
    ue = u in consumedNodes
    ve = v in consumedNodes
    if not (ue and ve):
        consumedEdges.append((u,v))
        if ue:
            print("{2} in, {0} not in N, tree={1}".format(v,consumedNodes[u],u))
            consumedNodes[v] = consumedNodes[u]
            trees[consumedNodes[v]].add(v)
        elif ve:
            print("{2} in, {0} not in N, tree={1}".format(u,consumedNodes[v],v))
            consumedNodes[u] = consumedNodes[v]
            trees[consumedNodes[u]].add(u)
        else:
            print("{0} and {1} not in N, tree={2}".format(u,v,currentTree))
            consumedNodes[u] = consumedNodes[v] = currentTree
            trees[currentTree] = set([u,v])
            currentTree+=1
    elif consumedNodes[u] != consumedNodes[v]:
        consumedEdges.append((u,v))
        newTree, oldTree = sorted([consumedNodes[u],consumedNodes[v]])
        print("{0} and {1} in N, trees={2},{3}".format(u, v, consumedNodes[u],consumedNodes[v]))
        print(trees[oldTree])
        x = set(trees[oldTree])
        for n in x:
            print("put {0} into {1}".format(n,newTree))
            trees[oldTree].remove(n)
            trees[newTree].add(n)
            consumedNodes[n] = newTree
    else:
        """
        neighbors = [(u,n,G[u][n]['weight']) for n in getNeighbors(u)]
        neighbors+= [(v,n,G[v][n]['weight']) for n in getNeighbors(v)]
        neighbors = [(u1,v1,d) for (u1,v1,d) in neighbors if (u1,v1)!=(u,v) and (v1,u1)!=(u,v)]
        n = max(neighbors,key=lambda n:n[2] if (n[0],n[1]) in consumedEdges or (n[1],n[0]) in consumedEdges else 0)
        ur,vr,dr = n

        if (vr,ur) in consumedEdges:
            consumedEdges.remove((vr,ur))
            print("replace ({0},{1}) (w={2}) with ({3},{4}) (w={5})".format(vr,ur,dr,u,v,d))
        if (ur,vr) in consumedEdges:
            consumedEdges.remove((ur,vr))
            print("replace ({0},{1}) (w={2}) with ({3},{4}) (w={5})".format(ur, vr, dr, u, v, d))
        consumedEdges.append((u, v))
        """

nx.draw_networkx_nodes(G, pos, node_size=10)
nx.draw_networkx_edges(G, pos, width=1)
nx.draw_networkx_edges(G, pos, edgelist=consumedEdges,
                       width=3, edge_color='r')
#elarge=[(u,v) for (u,v,d) in G.edges(data=True) if d['weight'] >0.5]
#esmall=[(u,v) for (u,v,d) in G.edges(data=True) if d['weight'] <=0.5]

#edgeLabels={(u,v):d['weight'] for (u,v,d) in G.edges(data=True)}
#nx.draw_networkx_edge_labels(G,pos,edge_labels=edgeLabels)

# labels
#nx.draw_networkx_labels(G,pos,font_size=5,font_family='sans-serif')

plt.axis('off')
plt.savefig("weighted_graph.png") # save as png
plt.show() # display
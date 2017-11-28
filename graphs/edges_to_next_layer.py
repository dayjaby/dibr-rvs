import numpy as np
import matplotlib
matplotlib.rcParams['text.usetex'] = True
matplotlib.rcParams['text.latex.unicode'] = True
matplotlib.rcParams['text.latex.preamble'] = [r'\usepackage{amsmath}', r'\usepackage{amssymb}']
import matplotlib.pyplot as plt
import networkx as nx

n = 3
m = 2
G=nx.DiGraph()
fig1 = plt.figure(1, (n+2+m*0.2,n+1+m*0.25))
ax = fig1.add_axes([0, 0, 1, 1], frameon=False, aspect=1.)
pos = {}
edgecolor = {}
vstart = (0)
vend = (m+1)
G.add_node(vstart)
G.add_node(vend)
#pos[vstart] = (0.2,n-0.5)
pos[vstart] = (0.2,1.25)
pos[vend] = (n+1+m*0.2,1.25)
for j in xrange(1,n+1):
    for i in xrange(1,m+1):
        for k in xrange(1,n+1):
            G.add_node((i,j,k))
            pos[(i,j,k)] = (k+i*0.2,n-j+0.25+(m-i)*0.25)
        G.add_edge((i,j,n),vend)
        edgecolor[((i,j,n),vend)] = 'blue'
        G.add_edge(vstart,(i,j,1))
        edgecolor[(vstart,(i,j,1))] = 'blue'
    #G.add_edge((m,j,n),vend)
    #edgecolor[((m,j,n),vend)] = 'blue'
for i in xrange(1,m+1):
    for j in xrange(1,n+1):
        for k in xrange(1,n):
            if i<m:
                for l in xrange(j+1,n+1):
                    u,v = (i,j,k),(i+1,l,k+1)
                    G.add_edge(u,v)
                    edgecolor[(u,v)] = 'green'
            u,v = (i,j,k),(i,j,k+1)
            G.add_edge(u,v)
            edgecolor[(u,v)] = 'black'
nx.draw_networkx_nodes(G, pos, node_size=10)
for u in G.nodes():
    px,py = pos[u]
    if u == vstart:
        label = r"$v_{\text{start}}$"
    elif u == vend:
        label = r"$v_{\text{end}}$"
    else:
        i,j,k = u
        label = r"$v_{{{},{},{}}}$".format(i,k,j)
    ax.text(px,py-0.15, label, fontsize=18)
        
        
for u,v in G.edges():
    ax.annotate("", pos[v],
                pos[u],
                #xycoords="figure fraction", textcoords="figure fraction",
                ha="right", va="center",
                arrowprops=dict(arrowstyle="-|>",
                                shrinkA=5,
                                shrinkB=5,
                                facecolor=edgecolor[(u,v)], edgecolor=edgecolor[(u,v)],
                                connectionstyle="arc3,rad=-0.05",
                                ),
                bbox=dict(boxstyle="square", fc="w"), )
ax.xaxis.set_visible(False)
ax.yaxis.set_visible(False)
plt.draw()
plt.axis('off')
plt.savefig("edges_to_next_layer.png")
plt.show() # display

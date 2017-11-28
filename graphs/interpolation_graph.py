import numpy as np
import matplotlib
matplotlib.rcParams['text.usetex'] = True
matplotlib.rcParams['text.latex.unicode'] = True
matplotlib.rcParams['text.latex.preamble'] = [r'\usepackage{amsmath}', r'\usepackage{amssymb}']
import matplotlib.pyplot as plt
import networkx as nx

n = 6
m = 4
G=nx.DiGraph()
fig1 = plt.figure(1, (n*0.8+0.5,m*0.6+0.5))
ax = fig1.add_axes([0, 0, 1, 1], frameon=False, aspect=1.)
pos = {}
edgecolor = {}
vstart = (1,1)
vend = (m,n)
G.add_node(vstart)
G.add_node(vend)
#pos[vstart] = (0.2,n-0.5)
pos[vstart] = (0.8,(m-1)*0.6)
pos[vend] = (n*0.8,0)
jrb = [1]
for i in xrange(2,m+1):
    if i == m:
        jr = [n]
    else:
        jr = xrange(i,n-m+i+1)
    for j in jr:
        G.add_node((i,j))
        pos[(i,j)] = (j*0.8,(m-i)*0.6)
        for j2 in jrb:
            if j2<j:
                G.add_edge((i-1,j2),(i,j))
                edgecolor[((i-1,j2),(i,j))] = 'black'
    jrb = jr
#for j2 in jrb:
#    G.add_edge((m-1,j2),(m,n))
#    edgecolor[((m-1,j2),(m,n))] = 'black'
    
nx.draw_networkx_nodes(G, pos, node_size=10)
        
        
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
for u in G.nodes():
    px,py = pos[u]
    i,j = u
    label = r"$v_{{{},{}}}$".format(i,j)
    ax.text(px,py, label, fontsize=18, bbox=dict(facecolor='none', edgecolor='black', fc='yellow'))
    
ax.xaxis.set_visible(False)
ax.yaxis.set_visible(False)
plt.draw()
plt.axis('off')
plt.savefig("interpolation_graph.png")
plt.show() # display

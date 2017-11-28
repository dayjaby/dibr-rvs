import numpy as np
import matplotlib
matplotlib.rcParams['text.usetex'] = True
matplotlib.rcParams['text.latex.unicode'] = True
matplotlib.rcParams['text.latex.preamble'] = [r'\usepackage{amsmath}', r'\usepackage{amssymb}']
import matplotlib.pyplot as plt
import networkx as nx

G=nx.DiGraph()
fig1 = plt.figure(1, (5,4))
ax = fig1.add_axes([0, 0, 0.95, 0.95], frameon=False, aspect=1.)
pos = {}
edgecolor = {}
vstart = vend = ()
x = ('i','j+1','r')
G.add_node(x)
pos[x] = (2,0)

G.add_node(('i','j','r'))
pos[('i','j','r')] = (1,0)
G.add_node(('i-1','j','r-1'))
pos[('i-1','j','r-1')] = (1,1)
G.add_node(('i-1','j','r-2'))
pos[('i-1','j','r-2')] = (1,2)
G.add_node(('i-1','j','i-1'))
pos[('i-1','j','i-1')] = (1,4)

G.add_edge(('i','j','r'),x)
edgecolor[(('i','j','r'),x)] = 'black'
G.add_edge(('i-1','j','r-1'),x)
edgecolor[(('i-1','j','r-1'),x)] = 'black'
G.add_edge(('i-1','j','r-2'),x)
edgecolor[(('i-1','j','r-2'),x)] = 'black'
G.add_edge(('i-1','j','i-1'),x)
edgecolor[(('i-1','j','i-1'),x)] = 'black'

x = ('i','j-1','r')
G.add_node(x)
pos[x] = (4,0)

G.add_node(('i','j','r '))
pos[('i','j','r ')] = (5,0)
G.add_node(('i+1','j','r+1'))
pos[('i+1','j','r+1')] = (5,1)
G.add_node(('i+1','j','r+2'))
pos[('i+1','j','r+2')] = (5,2)
G.add_node(('i+1','j','n'))
pos[('i+1','j','n')] = (5,4)

G.add_edge(x,('i','j','r '))
edgecolor[(x,('i','j','r '))] = 'black'
G.add_edge(x,('i+1','j','r+1'))
edgecolor[(x,('i+1','j','r+1'))] = 'black'
G.add_edge(x,('i+1','j','r+2'))
edgecolor[(x,('i+1','j','r+2'))] = 'black'
G.add_edge(x,('i+1','j','n'))
edgecolor[(x,('i+1','j','n'))] = 'black'



nx.draw_networkx_nodes(G, pos, node_size=10)
for u in G.nodes():
    px,py = pos[u]
    if u == vstart:
        label = r"$v_{\text{start}}$"
    elif u == vend:
        label = r"$v_{\text{end}}$"
    else:
        i,j,k = u
        label = r"$v_{{{},{},{}}}$".format(i,j,k)
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
ax.plot([1,1,1,5,5,5],[2.8,3,3.2,2.8,3,3.2],'ok',ms=2)
plt.draw()
plt.axis('off')
plt.savefig("proof_jr_boundaries_1.png")
plt.show() # display

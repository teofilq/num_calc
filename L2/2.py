import pickle
import numpy as np
import matplotlib.pyplot as plt
import networkx as nx
from networkx.algorithms import approximation

def metoda_puterii(A, tol=1e-9, max_iter=1000):
    n = A.shape[0]
    y = np.random.rand(n)
    y = y / np.linalg.norm(y)
    err = 1
    it = 0
    while err > tol and it < max_iter:
        z = A @ y
        z = z / np.linalg.norm(z)
        err = 1 - abs(z @ y)
        y = z
        it += 1
    valoare = (y @ (A @ y)) / (y @ y)
    return valoare, y

def is_complet(G):
    n = len(G.nodes())
    m = G.number_of_edges()
    return m == n*(n-1)//2

def main():
    # 1) n = 6
    A_test = np.random.rand(6,6)
    val_putere, vec_putere = metoda_puterii(A_test)
    val_eig, vec_eig = np.linalg.eig(A_test)
    print("Valoare proprie (Metoda Puterii):", val_putere)
    print("Valori proprii (numpy.linalg.eig):", val_eig)
    
    # 2
    with open('grafuri.pickle','rb') as f:
        A1, A2, A3 = pickle.load(f)
    
    for A in [A1, A2, A3]:
        G = nx.from_numpy_array(A)
        nx.draw(G, with_labels=True)
        plt.show()
        
        w, v = np.linalg.eig(A)
        print("Valorile proprii:", w)
        print("Este complet:", is_complet(G))
        print("Este bipartit:", nx.is_bipartite(G))
        c = approximation.clique.max_clique(G)
        print("Dimensiunea celei mai mari clici:", len(c))

if __name__ == '__main__':
    main()
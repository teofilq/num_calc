import numpy as np
import matplotlib.pyplot as plt
import pandas as pd
from sklearn.decomposition import PCA
import matplotlib.image as mpimg

def laborator3():
    # Ex1
    A = np.random.randn(10, 4)
    print("Rang inițial:", np.linalg.matrix_rank(A))
    C = A[:, :3] @ np.random.randn(3, 4)
    A_ext = np.hstack((A, C))
    print("Rang extins:", np.linalg.matrix_rank(A_ext))
    Z = np.random.normal(0, 0.2, A_ext.shape)
    A_zg = A_ext + Z
    print("Rang zgomot:", np.linalg.matrix_rank(A_zg))
    U, S, Vt = np.linalg.svd(A_zg, full_matrices=False)
    print("Valori singulare:", S)

    # Ex2
    img = mpimg.imread('imagine.png')
    if len(img.shape) == 3:
        img = img[:, :, 0]
    plt.imshow(img, cmap='gray')
    plt.show()
    U_i, S_i, Vt_i = np.linalg.svd(img, full_matrices=False)
    for k in [10, 50, 100]:
        img_k = U_i[:, :k] @ np.diag(S_i[:k]) @ Vt_i[:k, :]
        plt.imshow(img_k, cmap='gray')
        plt.show()

    # Ex3
    iris = pd.read_csv('iris.csv', header=None)
    X = iris.iloc[:, :4].values
    y = iris.iloc[:, 4].values
    pca = PCA(n_components=2)
    X_2 = pca.fit_transform(X)
    plt.scatter(X_2[:, 0], X_2[:, 1], c=pd.factorize(y)[0])
    plt.show()

def main():
    laborator3()

if __name__ == '__main__':
    main()
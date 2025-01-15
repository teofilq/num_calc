import numpy as np
import matplotlib.pyplot as plt
from skimage import io
from sklearn.feature_extraction.image import extract_patches_2d
from dictlearn import DictionaryLearning, methods
from sklearn.preprocessing import normalize

# Varianta din laborator folosea reconstruct_from_patches_2d, nu refăcea corect imaginea din patch-uri și ducea la Ic=0. 


def psnr(a,b):
    e=np.mean((a-b)**2)
    if e==0:return 0
    return 20*np.log10(255/np.sqrt(e))

def recm(Yc, shp, p):
    H,W=shp
    r=np.zeros((H,W))
    w=np.zeros((H,W))
    Y2=Yc.reshape(-1,p,p)
    idx=0
    for i in range(H-p+1):
        for j in range(W-p+1):
            r[i:i+p,j:j+p]+=Y2[idx]
            w[i:i+p,j:j+p]+=1
            idx+=1
    return r/(w+1e-12)

I=io.imread('imagine.png', as_gray=True).astype(float)

p=8
s=6
N=1000
n=256
K=50
sg=0.075

In=I+sg*np.random.randn(I.shape[0],I.shape[1])
Yn=extract_patches_2d(In,(p,p)).reshape(-1,p*p)
m=np.mean(Yn,axis=1,keepdims=True)
YnC=Yn-m
idx=np.random.choice(YnC.shape[0],N,replace=False)
Y=YnC[idx].T
D0=np.random.randn(Y.shape[0],n)
D0=normalize(D0,axis=0,norm='max')
dl=DictionaryLearning(n_components=n,max_iter=K,fit_algorithm='ksvd',n_nonzero_coefs=s,dict_init=D0,data_sklearn_compat=False)
dl.fit(Y)
D=dl.D_
X,_=methods.omp(YnC.T,D,n_nonzero_coefs=s)
Yc=(D@X).T+m
Yc=Yc.reshape(In.shape[0]-p+1,In.shape[1]-p+1,p*p)
Ic=recm(Yc,In.shape,p)
Ic=np.clip(Ic,0,1)

plt.subplot(131);plt.imshow(I,cmap='gray',vmin=0,vmax=1)
plt.subplot(132);plt.imshow(In,cmap='gray',vmin=0,vmax=1)
plt.subplot(133);plt.imshow(Ic,cmap='gray',vmin=0,vmax=1)


print(psnr(I,In), psnr(I,Ic))

plt.show()
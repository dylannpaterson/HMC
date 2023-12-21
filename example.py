import numpy as np
import matplotlib.pyplot as plt
from  hmcSampler import HMCSampler

def dU(q):


    return np.sum((y - f(q,x))*(-1.0*df(q,x)), axis=1)

def U(q):
    
    return np.sum((y - f(q,x))**2)/2.0

def f(q,x):
    return q[0]*x + q[1] 

def df(q,x):
    return np.asarray([x,1+x*0.0])

def main():
    global y
    global x

    x = np.linspace(-1.0,1.0,100)
    y = 2.0*x + 5.0 + np.random.randn(len(x))*0.2



    smp = HMCSampler()
    smp.n = 10000
    smp.qi = np.asarray([1.0,0.0])
    smp.HMC(U,dU)

    py = f((np.median(smp.sample,axis=1)),x)
    plt.plot(x,y,'.')
    plt.plot(x,py,'r-')
    plt.show()

    print(np.median(smp.sample[0,:]),np.median(smp.sample[1,:]))



if __name__ == "__main__":
    main()
    
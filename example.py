import numpy as np
import matplotlib.pyplot as plt
from  hmcSampler import HMCSampler

def dU(q):
    return np.sum((y - f(q,x))*(-1.0*df(q,x)), axis=1)

def U(q):
    return np.sum((y - f(q,x))**2)/2.0

def f(q,x):
    return np.polynomial.polynomial.polyval(x,q)
 
def df(q,x):
    indicies = np.arange(len(q))
    return np.power(np.asarray([x]*len(q)).T,indicies).T

def main():

    global y
    global x

    x = np.linspace(-5.0,5.0,100)
    p = np.asarray([2.0,5.0,-1.0,1.0])
    y = f(p,x) + np.random.randn(len(x))



    smp = HMCSampler()
    smp.tf = 0.1
    smp.qi = np.asarray([1.5,1.0,0.0,0.0])
    smp.HMC(U,dU)

    py = f((np.median(smp.sample,axis=1)),x)
    plt.plot(x,y,'.')
    plt.plot(x,py,'r-')
    plt.show()

    print(np.median(smp.sample,axis=1))

    plt.hist(smp.sample[0,:])
    plt.show()



if __name__ == "__main__":
    main()
    
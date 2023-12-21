import numpy as np
import matplotlib.pyplot as plt

def leapfrog (m,qi,pi,dU,t0,tf,e):

    t = np.arange(t0,tf,e)
    q = np.zeros((len(qi),len(t)))
    p = np.zeros((len(qi),len(t)))

    q[:,0] = qi
    p[:,0] = pi

    for i,ti in enumerate(t[:-1],start=1):
        ph = p[:,i-1] - (e/2.0)*dU(q[:,i-1])
        q[:,i] = q[:,i-1] + (e)*ph/m
        p[:,i] = ph - (e/2.0)*dU(q[:,i])


    return t,q,p

def dU(q):

    return q

def U(q):
    return np.sum(q**2)/2.0

def HMC(U,dU,q,e,tf,n):

    q_sample = np.zeros((len(q),n))
    q_sample[:,0] = q

    for ii in range(1,n):
        q = q_sample[:,ii-1]
        p = np.random.randn(len(q))
        t,qf,pf = leapfrog(1.0,q,p,dU,0.0,tf,e)

        qf = qf[:,-1]
        pf = -pf[:,-1]

        Ui  = U(q)
        Ki = sum(p**2)/2.0

        Uf = U(qf)
        Kf = sum(pf**2)/2.0

        deltaE = np.exp(Ui-Uf+Ki-Kf)
        acceptance = np.random.rand(1)

        q_sample[:,ii] = qf*(acceptance < deltaE) + q*(acceptance >= deltaE)

    return(q_sample)



def main():

    p0 = HMC(U,dU,np.random.randn(1),0.05,1.0,10000)


    plt.hist(p0[0,:],bins=30)
    plt.show()



if __name__ == "__main__":
    main()
    
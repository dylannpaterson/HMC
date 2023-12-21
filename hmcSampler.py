import numpy as np

class HMCSampler:
    def __init__(self):

        self.t0 = 0.0
        self.tf = 1.0
        self.e = 0.03
        self.n = 10000
        self.m = 1.0

    def leapfrog (self,qi,pi,U,dU):

        t = np.arange(self.t0,self.tf,self.e)
        q = np.zeros((len(self.qi),len(t)))
        p = np.zeros((len(self.qi),len(t)))

        q[:,0] = qi
        p[:,0] = pi

        for i,ti in enumerate(t[:-1],start=1):
            ph = p[:,i-1] - (self.e/2.0)*dU(q[:,i-1])
            q[:,i] = q[:,i-1] + (self.e)*ph/self.m
            p[:,i] = ph - (self.e/2.0)*dU(q[:,i])

        return t,q,p
    
    def HMC(self,U,dU):

        q_sample = np.zeros((len(self.qi),self.n))
        q_sample[:,0] = self.qi

        for ii in range(1,self.n):

            q = q_sample[:,ii-1]
            p = np.random.randn(len(q))
            t,qf,pf = self.leapfrog(q,p,U,dU)

            qf = qf[:,-1]
            pf = -pf[:,-1]

            Ui  = U(q)
            Ki = sum(p**2)/2.0

            Uf = U(qf)
            Kf = sum(pf**2)/2.0

            deltaE = np.exp(Ui-Uf+Ki-Kf)
            acceptance = np.random.rand(1)

            q_sample[:,ii] = qf*(acceptance < deltaE) + q*(acceptance >= deltaE)

            self.sample = q_sample
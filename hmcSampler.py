import numpy as np

class HMCSampler:
    def __init__(self):

        self.t0 = 0.0
        self.tf = 1.0
        self.e = 0.002
        self.m = 1.0

        self.n_walkers=20
        
        self.warn = ''

        self.n_samples = 10000
        self.n_burnin = 1000
        

    def leapfrog (self,qi,pi,U,dU):

        t = np.arange(self.t0,self.tf,self.e)
        q = np.zeros((len(self.qi),len(t),self.n_walkers))
        p = np.zeros((len(self.qi),len(t),self.n_walkers))

        q[:,0,:] = qi
        p[:,0,:] = pi

        for i,ti in enumerate(t[:-1],start=1):
            ph = p[:,i-1,:] - (self.e/2.0)*dU(q[:,i-1,:])
            q[:,i,:] = q[:,i-1,:] + (self.e)*ph/self.m
            p[:,i,:] = ph - (self.e/2.0)*dU(q[:,i,:])

        return t,q,p
    
    def HMC(self,U,dU):

        self.n = self.n_samples + self.n_burnin

        q_sample = np.zeros((len(self.qi),self.n,self.n_walkers))
        q_sample[:,0,:] = (self.qi + self.qi*np.random.randn(self.n_walkers,len(self.qi))).T

        for ii in range(1,self.n):

            q = q_sample[:,ii-1,:]
            p = np.random.randn(len(self.qi),self.n_walkers)
            t,qf,pf = self.leapfrog(q,p,U,dU)

            qf = qf[:,-1,:]
            pf = -pf[:,-1,:]



            Ui  = U(q)
            Ki = np.sum(p**2,axis=0)/2.0

            Uf = U(qf)
            Kf = np.sum(pf**2,axis=0)/2.0



            deltaE = np.exp(Ui-Uf+Ki-Kf)

            acceptance = np.random.rand(self.n_walkers)

            q_sample[:,ii,:] = (qf*(acceptance < deltaE) + q*(acceptance >= deltaE))

            

            if len(self.warn)>0:
                print(self.warn)

        self.samples = q_sample[:,self.n_burnin:,:].T
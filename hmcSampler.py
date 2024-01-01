import numpy as np

class HMCSampler:
    def __init__(self):

        self.t0 = 0.0
        self.steps = 50
        self.epsilon = 0.02
        self.m = 1.0

        self.p0 = 1.0

        self.n_walkers=20

        self.n_samples = 2000
        self.n_burnin = 1000

        self.warn = ''
        

    def leapfrog (self,qi,pi):

        t = np.arange(self.t0,self.t0+self.epsilon*self.steps,self.epsilon)
        q = np.zeros((len(self.qi),len(t),self.n_walkers))
        p = np.zeros((len(self.qi),len(t),self.n_walkers))

        q[:,0,:] = qi
        p[:,0,:] = pi

        for i,ti in enumerate(t[:-1],start=1):
            ph = p[:,i-1,:] - (self.epsilon/2.0)*self.dU(q[:,i-1,:])
            q[:,i,:] = q[:,i-1,:] + (self.epsilon)*ph/self.m
            p[:,i,:] = ph - (self.epsilon/2.0)*self.dU(q[:,i,:])

        return t,q,p
    
    def runHMC(self):

        self.n = self.n_samples + self.n_burnin

        q_sample = np.zeros((len(self.qi),self.n,self.n_walkers))
        q_sample[:,0,:] = (self.qi + self.qi*np.random.randn(self.n_walkers,len(self.qi))).T
        q_orbit = np.zeros((len(self.qi),self.steps,self.n,self.n_walkers))

        for ii in range(1,int(self.n/2)):

            q = q_sample[:,2*ii-1,:]
            p = self.p0*np.random.randn(len(self.qi),self.n_walkers)
            t,qf,pf = self.leapfrog(q,p)

            q_orbit[:,:,2*ii-1,:] = qf

            qf = qf[:,-1,:]
            pf = -pf[:,-1,:]

            Ui  = self.U(q)
            Ki = np.sum(p**2,axis=0)/2.0

            Uf_hmc = self.U(qf)
            Kf_hmc = np.sum(pf**2,axis=0)/2.0

            deltaE = np.exp(Ui-Uf_hmc+Ki-Kf_hmc)

            acceptance = np.random.rand(self.n_walkers)

            self.hmc_acceptance = acceptance < deltaE

            q_sample[:,2*ii,:] = (qf*(acceptance < deltaE) + q*(acceptance >= deltaE))

            qi_mcmc = q_sample[:,2*ii,:]

            median_delta_q = np.median(qf - q,axis=0)

            qf_mcmc = (qi_mcmc + median_delta_q*np.random.randn(len(self.qi),self.n_walkers))

            Uf_mcmc = self.U(qf_mcmc)

            deltaU = np.exp(Uf_hmc - Uf_mcmc)

            acceptance = np.random.rand(self.n_walkers)

            self.mcmc_acceptance = acceptance < deltaU

            q_sample[:,2*ii+1,:] = (qf_mcmc*(acceptance < deltaU) + qf*(acceptance >= deltaU))

            if len(self.warn)>0:
                print(self.warn)

        self.samples = q_sample[:,self.n_burnin:,:].T
        self.orbits = q_orbit[:,:,self.n_burnin:,:].T
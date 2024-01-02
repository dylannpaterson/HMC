import numpy as np
import matplotlib.pyplot as plt
import corner
from scipy.integrate import simpson
from matplotlib.ticker import FormatStrFormatter

class HMCSampler:
    def __init__(self):

        self.t0 = 0.0
        self.steps = 20
        self.lf_length = 1.0
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
        self.n_parameters = len(self.qi)

        self.epsilon = self.lf_length/self.steps

        q_sample = np.zeros((len(self.qi),self.n,self.n_walkers))
        q_sample[:,0,:] = (self.qi + self.qi*np.random.randn(self.n_walkers,len(self.qi))).T
        q_orbit = np.zeros((len(self.qi),self.steps,self.n,self.n_walkers))


        for ii in range(1,int(self.n)):

            q = q_sample[:,ii-1,:]
            p = self.p0*np.random.randn(len(self.qi),self.n_walkers)
            t,qf,pf = self.leapfrog(q,p)

            q_orbit[:,:,ii-1,:] = qf

            qf = qf[:,-1,:]
            pf = -pf[:,-1,:]

            Ui  = self.U(q)
            Ki = np.sum(p**2,axis=0)/2.0

            Uf_hmc = self.U(qf)
            Kf_hmc = np.sum(pf**2,axis=0)/2.0

            deltaE = np.exp(Ui-Uf_hmc+Ki-Kf_hmc)

            acceptance = np.random.rand(self.n_walkers)

            self.hmc_acceptance = acceptance < deltaE

            q_sample[:,ii,:] = (qf*(acceptance < deltaE) + q*(acceptance >= deltaE))

            if len(self.warn)>0:
                print(self.warn)

        self.samples = np.reshape(q_sample[:,self.n_burnin:,:].T,(self.n_samples*self.n_walkers,len(self.qi)))

        self.orbits = np.reshape(q_orbit[:,:,self.n_burnin:,:].T,(self.n_samples*self.n_walkers,self.steps,len(self.qi)))

    def plotSamples(self, labels):
        figure = corner.corner(
            self.samples,
            labels=labels,
                quantiles=[0.16, 0.5, 0.84],
                show_titles=True,
                title_kwargs={"fontsize": 12})
        plt.show()

    def plotOrbits(self, n_orbits):

        fig,ax = plt.subplots(self.n_parameters, self.n_parameters, sharex= 'col', 
                              figsize=(3.0*self.n_parameters,3.0*self.n_parameters))

        n = 30

        ps = []
        for i in range(self.n_parameters):
            ps.append(np.linspace(np.min(self.samples[:,i]),np.max(self.samples[:,i]),n))


        pp = np.asarray(np.meshgrid(*ps, indexing='ij'))

        uu = np.exp(-self.U(pp.reshape((self.n_parameters,n**self.n_parameters)))).reshape([n]*self.n_parameters)

        for i in range(self.n_parameters):
            for j in range(self.n_parameters):

                if i == j:

                    axes = {i,j}

                    other_axes = list(set(range(self.n_parameters)) - axes)
                    other_axes.reverse()

                    cntr = uu
                    px = pp*1.0
                    ppx = pp[i]
                    ppy = pp[j]


                    for k in other_axes:
                        cntr = simpson(cntr,px[k,...],axis=k)
                        cntr = cntr/np.sum(cntr)
                        px = np.min(px,axis=k+1)

                        ppx = np.min(ppx,axis=k)
                        ppy = np.min(ppy,axis=k)

                    ax[j,i].plot(ppx,cntr, 'k-')
                    ax[j,i].set_xticks(np.linspace(np.min(ppx),np.max(ppx),5))
                    ax[j,i].xaxis.set_major_formatter(FormatStrFormatter('%.2f'))
                    ax[j,i].set_yticklabels ([])


                elif i<j:

                    axes = {i,j}

                    other_axes = list(set(range(self.n_parameters)) - axes)
                    other_axes.reverse()

                    cntr = uu
                    px = pp
                    ppx = pp[i]
                    ppy = pp[j]


                    for k in other_axes:
                        cntr = simpson(cntr,px[k,...],axis=k)
                        px = np.min(px,axis=k+1)

                        ppx = np.min(ppx,axis=k)
                        ppy = np.min(ppy,axis=k)

                    ax[j,i].contourf(ppx,ppy,cntr, levels = 10, norm='linear', cmap = 'Blues', alpha=0.5)
                    ax[j,i].set_xticks(np.linspace(np.min(ppx),np.max(ppx),5))
                    ax[j,i].set_yticks(np.linspace(np.min(ppy),np.max(ppy),5))
                    ax[j,i].yaxis.set_major_formatter(FormatStrFormatter('%.2f'))
                    ax[j,i].plot(self.orbits[:n_orbits,:,i].T,self.orbits[:n_orbits,:,j].T,'k-')

                    if i>0:
                        ax[j,i].set_yticklabels ([])

                else:
                    ax[j,i].axis('off')

        plt.show()
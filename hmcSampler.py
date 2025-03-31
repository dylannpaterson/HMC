import numpy as np
import matplotlib.pyplot as plt
import corner
from scipy.integrate import simpson
from scipy.ndimage import gaussian_filter
from scipy.stats import gaussian_kde
from matplotlib.ticker import FormatStrFormatter
from tqdm import tqdm
import pickle
from datetime import datetime


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

        self.state= None
        self.chains= None
        self.samples = None
        self.orbits = None
        self.acceptance = None

        self.dU = self.dU_numerical
        self.dU_step_size = 1e-9

        self.warn = ''

        self.file_name = None
        self.save_every = None
        

    def leapfrog (self,qi,pi):
        """
        Perform a Leapfrog integration for simulating Hamiltonian dynamics.

        The Leapfrog method is commonly used in Molecular Dynamics and 
        Hamiltonian Monte Carlo simulations for solving differential equations 
        describing particle motion. It updates position (`q`) and momentum (`p`) 
        in an alternating manner with time (`t`).

        Parameters
        ----------
        qi : numpy.ndarray
            Shape (n_parameters, n_particles).
            Initial positions of the particles. It should have dimensions 
            corresponding to the number of particles and spatial dimensions.
        pi : numpy.ndarray
            Initial momenta of the particles. It should have dimensions 
            matching those of `qi`.

        Returns
        -------
        t : numpy.ndarray
            Array of time steps.
        q : numpy.ndarray
            Array of positions at each time step for all particles.
        p : numpy.ndarray
            Array of momenta at each time step for all particles.

        Notes
        -----
        - `self.dU(q)` should compute the gradient of the potential energy 
          with respect to `q`.
        - The timestep (`self.epsilon`) and number of steps (`self.steps`) 
          control the resolution and duration of the simulation.
        - Particle mass (`self.m`) affects how momentum (`p`) and position (`q`) 
          are updated.
        - This implementation assumes `self.n_walkers` determines the number 
          of independent trajectories being simulated.

        Example
        -------
        t, q, p = obj.leapfrog(initial_positions, initial_momenta)
        """

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
    
    def initialise_walkers(self):
        """
        Initialize the walkers for Hamiltonian Monte-Carlo.

        This function sets up the initial conditions for the walkers, including 
        their positions (`q_sample`), orbital paths (`q_orbit`), and acceptance 
        metrics (`q_acceptance`). Walkers are a collection of independent entities 
        that evolve through the parameter space using Hamiltonian mechanics.

        Returns
        -------
        q_sample : numpy.ndarray
            Shape (n_parameters, n_collect, n_walkers). Contains the initial positions 
            of the walkers. If `self.state` is None, positions are randomly initialized 
            around `self.qi` with a small Gaussian perturbation. Otherwise, it uses the 
            provided state.
        q_orbit : numpy.ndarray
            Shape (n_parameters, steps, n_collect, n_walkers). Pre-allocated array to 
            store the orbital paths of the walkers over the simulation steps.
        q_acceptance : numpy.ndarray
            Shape (n_collect, n_walkers). Array initialized to ones, representing the 
            acceptance metrics of walkers during the simulation.

        Notes
        -----
        - `self.qi` represents the initial positions used for initialization.
        - `self.n_collect` is the number of collection intervals for the simulation.
        - `self.n_walkers` is the total number of walkers in the simulation.
        - `self.steps` defines the number of steps in the simulation.
        - `self.state`, if provided, is used as the initial state of the walkers.

        Example
        -------
        q_sample, q_orbit, q_acceptance = obj.initialise_walkers()
        """

        q_sample = np.zeros((len(self.qi),self.n_collect,self.n_walkers))

        if self.state is None:
            q_sample[:,0,:] = (self.qi + 0.01*np.random.randn(self.n_walkers,len(self.qi))).T
        else:
            q_sample[:,0,:] = self.state

        q_orbit = np.zeros((len(self.qi),self.steps,self.n_collect,self.n_walkers))

        q_acceptance = np.ones((self.n_collect,self.n_walkers))
        
        return q_sample, q_orbit, q_acceptance
    
    def initialise_parameters(self):
        """
        Initialize the parameters required for the Hamiltonian Monte-Carlo.

        This function calculates and sets key attributes necessary for the 
        execution of the simulation, such as the total number of iterations, 
        collection intervals, number of parameters, step size, and initial 
        state values.

        Notes
        -----
        - If `self.state` is None, a new simulation run is initialized with a total 
          number of iterations (`self.n`) equal to the sum of samples and burn-in steps.
          The number of collection intervals (`self.n_collect`) is also initialized to 
          match this total.
        - If `self.state` is provided, the function updates the simulation parameters 
          to collect additional samples while maintaining the existing state.
        - The step size (`self.epsilon`) is computed based on the specified 
          leapfrog path length (`self.lf_length`) and the number of steps (`self.steps`).

        Attributes Set
        --------------
        self.n : int
            Total number of iterations (samples + burn-in steps).
        self.n_collect : int
            Total number of collection intervals during the simulation.
        self.n_parameters : int
            Number of parameters (determined by the length of `self.qi`).
        self.epsilon : float
            Step size for the leapfrog integration, calculated as 
            `self.lf_length / self.steps`.
        self.initial_chains : numpy.ndarray
            Stores the initial state of the chains for the simulation.
        self.initial_acceptance : numpy.ndarray
            Stores the initial acceptance rates for the walkers.

        Example
        -------
        obj.initialise_parameters()
        """

        if self.state is None:
            self.n = self.n_samples + self.n_burnin
            self.n_collect = self.n
        else:
            self.n += self.n_samples
            self.n_collect = self.n_samples + 1

        self.n_parameters = len(self.qi)

        self.epsilon = self.lf_length/self.steps

        self.initial_chains = self.chains
        self.initial_acceptance = self.acceptance

        
    def generate_sample(self,q):
        """
        Generate a new sample in phase space for the walkers.

        This function performs Hamiltonian Monte Carlo (HMC) sampling for a given
        set of initial positions (`q`). Each walker is assigned a random momentum, 
        and the phase space is explored using the Leapfrog integration method. The 
        function evaluates energy changes to determine whether new positions are 
        accepted or rejected, ensuring detailed balance in the sampling process.

        Parameters
        ----------
        q : numpy.ndarray
            Shape (n_parameters, n_walkers). The current positions of the walkers.

        Returns
        -------
        qf : numpy.ndarray
            Shape (n_parameters, n_walkers). The final positions of the walkers after 
            the sampling step, with accepted proposals applied.
        orbit : numpy.ndarray
            Shape (n_parameters, n_time_steps, n_walkers). The trajectory of each walker
            through phase space during the sampling step.
        acceptance : numpy.ndarray
            Shape (n_walkers,). A boolean array indicating whether the new position 
            was accepted (`True`) or rejected (`False`) for each walker.

        Notes
        -----
        - The momenta of the walkers are initialized with a Gaussian distribution scaled 
          by `self.p0`.
        - The `self.leapfrog` method is used to perform the integration of motion through 
          phase space.
        - The potential energy (`U`) and kinetic energy (`K`) are used to compute the 
          Hamiltonian dynamics and energy conservation for proposal acceptance.
        - Steps with energy changes (`deltaE`) greater than a random value are accepted; 
          otherwise, the walker remains in its previous position.

        Example
        -------
        qf, orbit, acceptance = obj.generate_sample(current_positions)
        """

        #give each walker a random momentum
        p = self.p0*np.random.randn(len(self.qi),self.n_walkers)

        #let each walker traverse the phase space based on initial position and momentum
        t,qf,pf = self.leapfrog(q,p)

        #save the path that each walker traversed
        orbit = qf*1.0

        #save the final momentum and position of each walker after traversing the phase space
        qf = qf[:,-1,:]
        pf = -pf[:,-1,:]

        #determine initial potential energy and kinetic energy
        Ui  = self.U(q)
        Ki = np.sum(p**2,axis=0)/2.0

        #determine final potential energy and kinetic energy
        Uf_hmc = self.U(qf)
        Kf_hmc = np.sum(pf**2,axis=0)/2.0

        #determine change in energy for each walker

        deltaE = np.exp(Ui-Uf_hmc+Ki-Kf_hmc)

        #accept new position if change in energy is greater than some random number between 0 and 1
        #Ideally, energy is conserved and deltaE ~ 1, so steps are nearly always accepted
        #if not accepted, walker stays in initial position
        acceptance = np.random.rand(self.n_walkers)

        return (qf*(acceptance < deltaE) + q*(acceptance >= deltaE)), orbit, acceptance < deltaE

    def save_chains(self,q_sample):
        """
        Save or update the Markov chains with the newly sampled positions.

        This function manages the storage of sampled positions (`q_sample`) by either 
        initializing the chains or appending new samples to the existing chains. 

        Parameters
        ----------
        q_sample : numpy.ndarray
            Shape (n_parameters, n_collect, n_walkers). The new sample positions 
            generated during the simulation.

        Notes
        -----
        - If no initial chains are present (`self.initial_chains` is None), the new 
          sample positions are assigned directly to `self.chains`.
        - If initial chains already exist, the function appends the new samples 
          (excluding the first collection interval to avoid duplication) to the existing chains 
          along the collection axis.

        Attributes Modified
        -------------------
        self.chains : numpy.ndarray
            The updated Markov chains containing all the sampled positions.

        Example
        -------
        obj.save_chains(new_sample)
        """
        if self.initial_chains is None:
            self.chains = q_sample
        else:
            self.chains = np.concatenate((self.initial_chains,q_sample[:,1:,:]),axis=1)

    def save_acceptance(self,q_acceptance):
        """
        Save or update the acceptance rates of the walkers.

        This function manages the storage of acceptance data (`q_acceptance`) by either 
        initializing the acceptance array or appending new acceptance values to the 
        existing data. 

        Parameters
        ----------
        q_acceptance : numpy.ndarray
            Shape (n_collect, n_walkers). The new acceptance data generated during 
            the simulation.

        Notes
        -----
        - If no initial acceptance data is present (`self.initial_acceptance` is None), 
          the function initializes `self.acceptance` with the provided `q_acceptance`.
        - If initial acceptance data exists, the function appends new acceptance values 
          (excluding the first collection interval to avoid duplication) to the existing 
          acceptance data along the collection axis.

        Attributes Modified
        -------------------
        self.acceptance : numpy.ndarray
            The updated array containing all recorded acceptance rates.

        Example
        -------
        obj.save_acceptance(new_acceptance)
        """
        if self.initial_acceptance is None:
            self.acceptance = q_acceptance
        else:
            self.acceptance = np.concatenate((self.initial_acceptance,q_acceptance[1:,:]),axis=0)

    def save_samples(self):
        """
        Extract and save the post-burn-in samples from the Markov chains.

        This function processes the stored Markov chains to retrieve the samples 
        collected after the burn-in period (`self.n_burnin`). The post-burn-in 
        samples are reshaped for further analysis or use in statistical evaluations. 
        If the chains contain fewer iterations than the burn-in period, no samples 
        are saved.

        Notes
        -----
        - The burn-in period (`self.n_burnin`) is used to discard initial samples 
          that may not represent the equilibrium distribution.
        - The reshaping operation converts the samples into a 2D array where each 
          row represents a single sample, and each column represents a parameter.

        Attributes Modified
        -------------------
        self.samples : numpy.ndarray or None
            If valid samples exist, a 2D array of shape 
            (n_samples_post_burnin, n_parameters) is stored. If no valid samples 
            are available (i.e., `self.chains` contains fewer iterations than 
            `self.n_burnin`), `self.samples` is set to `None`.

        Example
        -------
        obj.save_samples()
        """
        if self.chains.shape[1] > self.n_burnin:
            self.samples = self.chains[:,self.n_burnin:,:].T
            self.samples = np.reshape(self.samples,(self.samples.shape[0]*self.samples.shape[1],self.n_parameters))
        else:
            self.samples = None

    def save_orbits(self, q_orbit):
        """
        Save or update the orbital paths of the walkers.

        This function processes and stores the orbital paths (`q_orbit`) generated 
        during the simulation. If no orbits are currently saved (`self.orbits` is None), 
        it initializes `self.orbits` by reshaping the input orbital data to a 3D array. 
        If orbits already exist, the function appends the new orbital data to the 
        existing array along the sample axis.

        Parameters
        ----------
        q_orbit : numpy.ndarray
            Shape (n_dimensions, steps, n_collect, n_walkers). The orbital paths of the 
            walkers during the simulation.

        Notes
        -----
        - The function discards the burn-in phase (`self.n_burnin`) when initializing 
          or appending orbital paths to ensure only valid samples are stored.
        - Orbital paths are reshaped into a 3D array with dimensions 
          (n_samples * n_walkers, steps, n_dimensions) for efficient storage.
        - When appending new data, the first collection interval is excluded to 
          avoid duplication.

        Attributes Modified
        -------------------
        self.orbits : numpy.ndarray
            A 3D array of shape (n_total_samples * n_walkers, steps, n_dimensions) 
            storing all recorded orbital paths.

        Example
        -------
        obj.save_orbits(new_orbital_data)
        """
        if self.orbits is None:
            self.orbits = np.reshape(q_orbit[:,:,self.n_burnin:,:].T,(self.n_samples*self.n_walkers,self.steps,len(self.qi)))
        else:
            self.orbits = np.concatenate((self.orbits,np.reshape(q_orbit[:,:,1:,:].T,(self.n_samples*self.n_walkers,self.steps,len(self.qi)))),axis=0)

    def run_hmc(self):
        """
        Execute the Hamiltonian Monte Carlo (HMC) sampling process.

        This function orchestrates the entire HMC workflow, from initializing 
        parameters and walkers to performing sampling, saving intermediate results, 
        and updating the final states. It uses a progress bar to track the sampling 
        process and saves results at specified intervals.

        Notes
        -----
        - The sampling process involves iteratively generating new samples, recording 
          their orbital paths, and tracking acceptance rates for each walker.
        - If warnings are generated during sampling (`self.warn`), they are displayed 
          and reset.
        - Intermediate results (chains, samples, acceptance rates) are saved at 
          regular intervals if `self.save_every` is specified.

        Workflow
        --------
        1. Initializes parameters and walkers.
        2. Iteratively generates samples, orbits, and acceptance rates for the walkers.
        4. Saves intermediate results at specified intervals.
        5. Finalizes and saves all results at the end of the process.

        Attributes Modified
        -------------------
        self.chains : numpy.ndarray
            Updated Markov chains containing all collected samples.
        self.samples : numpy.ndarray
            Post-burn-in samples reshaped for analysis.
        self.acceptance : numpy.ndarray
            Updated acceptance rates for all iterations.
        self.orbits : numpy.ndarray
            Orbital paths of walkers collected during the simulation.
        self.state : numpy.ndarray
            The final state of the walkers at the end of the sampling process.

        Example
        -------
        obj.run_hmc()
        """
        self.initialise_parameters()

        q_sample, q_orbit, q_acceptance = self.initialise_walkers()

        #this controls the progress bar
        with tqdm(total=self.n_collect, desc="Collecting samples") as pbar:
            #for each sample to be collected
            for ii in range(1,self.n_collect):
                
                q_sample[:,ii,:], q_orbit[:,:,ii-1,:],q_acceptance[ii,:]  = self.generate_sample(q_sample[:,ii-1,:])

                if len(self.warn)>0:
                    print(self.warn)
                    self.warn = ''

                if self.save_every is not None and ii>1 and ii % self.save_every==0:
                    self.save_chains(q_sample[:,:ii,:])
                    self.save_samples()
                    self.save_acceptance(q_acceptance[:ii,:])
                    self.state = q_sample[:,ii,:]
                    self.save()


                pbar.update(1)

        self.save_chains(q_sample)

        self.save_samples()

        self.save_acceptance(q_acceptance)

        self.save_orbits(q_orbit)

        self.state = q_sample[:,-1,:]

    def plot_samples(self, labels: list=None):
        """Produces corner plot of samples

        Args:
            labels (list): names of parameters to be added to plots

        Returns:
            matplotlib.figure.Figure: corner plot of samples
        """
        if labels is None:
            labels = self.dim_labels
        
        figure = corner.corner(
            self.samples,
            labels=labels,
                quantiles=[0.16, 0.5, 0.84],
                show_titles=True,
                title_kwargs={"fontsize": 12})
        return figure

    def plot_orbits(self, n_orbits: int=10, labels: list=None):
        """Produces corner plot with first n_orbits orbits overplotted

        Args:
            n_orbits (int): number of orbits to be displayed on corner plot
            labels (list): names of parameters to be added to plots
        Returns:
            matplotlib.figure.Figure: corner plot of samples with obrits overplotted
        """        ''''''

        figure,ax = plt.subplots(self.n_parameters, self.n_parameters, sharex= 'col', 
                              figsize=(2.5*self.n_parameters,2.5*self.n_parameters))

        if labels is None:
            labels = self.dim_labels

        for i in range(self.n_parameters):
            for j in range(self.n_parameters):

                if i == j:

                    hist, xedges = np.histogram(self.samples[:,i],51)

                    xcentres = (xedges[1:] + xedges[:-1])/2.0


                    ax[j,i].hist(self.samples[:,i], 30, histtype ='step', color='k')
                    ax[j,i].set_xticks(np.linspace(np.min(xcentres),np.max(xcentres),5))
                    ax[j,i].xaxis.set_major_formatter(FormatStrFormatter('%.2f'))
                    ax[j,i].set_yticklabels ([])

                    low,med,hi = np.quantile(self.samples[:,i],(0.16,0.5,0.84))

                    low = low - med
                    hi = hi - med

                    ax[j,i].set_title(labels[i] + '={:.2f}'.format(med) + '$_{{{0:.2f}}}^{{+{1:.2f}}}$'.format(low,hi))

                    lowy,hiy = ax[j,i].get_ylim()

                    ax[j,i].plot([med,med],[1e-24,1e24],'k--')
                    ax[j,i].plot([med+low,med+low],[1e-24,1e24],'k:')
                    ax[j,i].plot([med+hi,med+hi],[1e-24,1e24],'k:')

                    lowx, hix = np.quantile(self.samples[:,i],(0.001,0.999))

                    ax[j,i].set_xlim((lowx,hix))
                    ax[j,i].set_ylim((lowy,hiy))

                    if j == self.n_parameters-1:
                        ax[j,i].set_xlabel(labels[i])

                    if i == 0:
                        ax[j,i].set_ylabel(labels[j])


                elif i<j:

                    axes = {i,j}

                    other_axes = list(set(range(self.n_parameters)) - axes)
                    other_axes.reverse()

                    xcentres = np.linspace(np.min(self.samples[:,i]),np.max(self.samples[:,i]),51)
                    ycentres = np.linspace(np.min(self.samples[:,j]),np.max(self.samples[:,j]),51)

                    xx,yy = np.meshgrid(xcentres,ycentres)

                    kern = gaussian_kde(np.vstack([self.samples[:,i],self.samples[:,j]]))

                    hist = np.reshape(kern(np.vstack([xx.ravel(),yy.ravel()])).T, xx.shape)
                    hist = hist/np.max(hist)

                    medx = np.quantile(self.samples[:,i],0.5)
                    medy = np.quantile(self.samples[:,j],0.5)

                    lowx, hix = np.quantile(self.samples[:,i],(0.001,0.999))
                    lowy, hiy = np.quantile(self.samples[:,j],(0.001,0.999))

                    lvls = [0.118, 0.393, 0.675,0.864,1.0]

                    ax[j,i].contourf(xcentres,ycentres,hist, levels = lvls, norm='linear', cmap = 'Greys', alpha=0.7)

                    ax[j,i].plot(xcentres,medy*(xcentres*0.0 + 1.0),'k--', alpha = 0.5)
                    ax[j,i].plot(medx*(xcentres*0.0 + 1.0),ycentres,'k--', alpha = 0.5)

                    ax[j,i].plot(self.orbits[:n_orbits,:,i].T,self.orbits[:n_orbits,:,j].T,'b-', alpha = 0.7)
                    ax[j,i].plot(self.samples[:n_orbits+1,i],self.samples[:n_orbits+1,j],'bo', alpha = 0.7)

                    ax[j,i].set_xlim((lowx,hix))
                    ax[j,i].set_ylim((lowy,hiy))
                    ax[j,i].set_xticks(np.linspace(lowx,hix,5))
                    ax[j,i].set_yticks(np.linspace(lowy,hiy,5))
                    ax[j,i].yaxis.set_major_formatter(FormatStrFormatter('%.2f'))


                    if j == self.n_parameters-1:
                        ax[j,i].set_xlabel(labels[i])

                    if i == 0:
                        ax[j,i].set_ylabel(labels[j])

                    if i>0:
                        ax[j,i].set_yticklabels ([])


                else:
                    ax[j,i].axis('off')

        plt.tight_layout(pad=0.4, w_pad=0.1, h_pad=0.1)
        return figure
    
    def plot_chains(self,labels: list = None):

        if labels is None:
            labels = self.dim_labels

        figure,ax = plt.subplots(self.n_parameters+1, sharex= 'col', 
        figsize=(4.5,1.5*self.n_parameters))

        for i in range(self.n_parameters):
            if(self.samples is not None):
                p16,p50,p84 = np.percentile(self.samples[:,i],[16,50,84])
                ax[i].plot([0,self.n],[p16,p16],'r-.')
                ax[i].plot([0,self.n],[p50,p50],'r--')
                ax[i].plot([0,self.n],[p84,p84],'r-.')

            ax[i].plot(self.chains[i,:,:],'k',alpha=0.1)
            yl = ax[i].get_ylim()
            ax[i].plot([self.n_burnin,self.n_burnin],[-1e6,1e6],'b--')
            ax[i].set_ylabel(labels[i])
            ax[i].set_xlim([-1,self.n+1])
            ax[i].set_ylim([yl[0],yl[1]])

        ax[self.n_parameters].plot(self.U(self.chains), 'k',alpha=0.1)
        ax[self.n_parameters].set_ylabel('U')
        ax[self.n_parameters].set_xlim([-1,self.n+1])


        ax[self.n_parameters].set_xlabel('Iteration')
        plt.tight_layout(pad=0.4, w_pad=0.1, h_pad=0.1)
        return figure
    
    def dU_numerical(self,q):

        output = np.zeros(q.shape)

        U0 = self.U(q)
        q0 = q*0.0

        steps = q.shape[0]

        for i in range(steps):
            qi = q0*0.0
            qi[i] = self.dU_step_size
    
            U1 = self.U(q+qi)
            #U2 = self.U(q-qi)

            output[i] = (U1-U0)/(self.dU_step_size)


        return output
    
    def reset(self):
        """
        Reset all simulation attributes.

        This function clears the stored state, chains, samples, orbits, and 
        acceptance data, effectively preparing the object for a fresh simulation 
        run.

        Attributes Reset
        -----------------
        self.state : None
            The current state of the walkers is cleared.
        self.chains : None
            The Markov chains are cleared.
        self.samples : None
            The collected samples are cleared.
        self.orbits : None
            The recorded orbital paths are cleared.
        self.acceptance : None
            The acceptance rates are cleared.

        Example
        -------
        obj.reset()
        """
        self.state = None
        self.chains = None
        self.samples = None
        self.orbits = None
        self.acceptance = None

    def save(self, file_name: str=None,add_time = True):
        """
        Save the current state of the object to a file.

        This function serializes the current state of the object and saves it 
        to a file using Python's `pickle` module. The file name can be specified 
        manually or generated automatically, with an option to include a timestamp 
        for uniqueness.

        Parameters
        ----------
        file_name : str, optional
            The base name of the file to save the object to. If not provided, the 
            default name `'HMC'` is used. If `self.file_name` is set, it will override 
            this default.
        add_time : bool, optional
            If `True`, appends the current date and time to the file name to ensure 
            uniqueness. Default is `True`.

        Notes
        -----
        - If the file name ends with the extension `.hmc`, the extension is removed 
          before generating the final file name.
        - The timestamp format is `YYYY-MM-DD-HH-MM-SS`.
        - The file is saved with the `.hmc` extension by default if not already present.
        - This method uses `pickle` for serialization, so any object attributes must 
          be serializable.

        Example
        -------
        obj.save("my_simulation", add_time=True)
        """
        if file_name is None and self.file_name is None:
            file_name = 'HMC'

        if file_name is None and self.file_name is not None:
            file_name = self.file_name

        if file_name.endswith('.hmc'):
            file_name = file_name[:-4]

        if add_time:
            now = datetime.now().strftime("%Y-%m-%d-%H-%M-%S")
            file_name += '_'+now

        if not file_name.endswith('.hmc'):
            file_name += '.hmc' 

        with open(file_name, "wb") as file:
            pickle.dump(self, file)
    
    def load(self, file_name: str):
        """
        Load a previously saved object state from a file and update the current instance.

        This function deserializes a saved object using Python's `pickle` module 
        and transfers its attributes to the current instance. It ensures that only 
        non-private, non-callable attributes are copied, preserving methods and private 
        attributes of the existing instance.

        Parameters
        ----------
        file_name : str
            The name of the file from which the object state is to be loaded. 
            The file must have been saved using a compatible method (e.g., `save`).

        Notes
        -----
        - Private attributes (those starting with `"__"`) and methods are excluded 
          when transferring attributes from the loaded object.
        - The loaded attributes overwrite any existing attributes in the current instance 
          with the same name.
        - This method relies on `pickle`, so the file must contain a serialized object 
          compatible with the current class.

        Example
        -------
        obj.load("saved_simulation.hmc")
        """
        with open(file_name, "rb") as file:
            loaded = pickle.load(file)

        for attr in dir(loaded):
            # Filter out private attributes and methods
            if not attr.startswith("__") and not callable(getattr(loaded, attr)):
                value = getattr(loaded, attr)  # Get the attribute value
                setattr(self, attr, value) 


        

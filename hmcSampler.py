import os
# This flag must be set before jax is imported.
# Set this to the number of CPU cores you want JAX to see.
# If you have a GPU, JAX will use it by default and this flag will be ignored.
os.environ['XLA_FLAGS'] = '--xla_force_host_platform_device_count=8'

import jax
import jax.numpy as jnp
from jax import random
import numpy as np # Retain numpy for plotting and scipy compatibility
import matplotlib.pyplot as plt
from matplotlib.patches import Ellipse
import corner
from scipy.stats import gaussian_kde
from matplotlib.ticker import FormatStrFormatter
from tqdm import tqdm
import pickle
from datetime import datetime
import time
import functools


class HMCSampler:
    """
    Hamiltonian Monte Carlo (HMC) sampler refactored to use JAX.

    This sampler leverages JAX for automatic differentiation to compute gradients
    of the potential energy function `U` by default. It can also accept a
    manually provided gradient function `dU`.

    The core sampling step is parallelized across available devices using `jax.pmap`.
    """

    def __init__(self, U, dU=None):
        """
        Initializes the HMC sampler.

        The core logic for selecting the gradient function resides here.

        Parameters:
            U (callable): The potential energy function (negative log probability).
                          This function must be JAX-compatible and operate on a
                          single position vector of shape (n_parameters,).
            dU (callable, optional): The gradient of the potential energy. Must also
                                     operate on a single position vector. If None,
                                     `jax.grad(U)` is used by default.
        """
        # --- Core Sampler Attributes ---
        self.qi = None # The initial center point for the walkers
        self.steps = 64
        self.lf_length = 1.0
        self.m = 1.0
        self.p0 = 1.0
        self.n_walkers = 64
        self.n_samples = 2000
        self.n_burnin = 1000
        self.dim_labels = None # For plotting
        self.store_orbits = False # Set to True to enable orbit plotting

        # --- State and Results ---
        self.state = None
        self.chains = None
        self.samples = None
        self.orbits = None
        self.acceptance = None
        self.warn = ''

        # --- File I/O ---
        self.file_name = None
        self.save_every = None

        # --- THE STRATEGIC CHOICE: JAX Auto-Grad or Manual ---
        U_single = U
        dU_single = dU

        if dU_single is None:
            print("`dU` not supplied. Using `jax.grad(U)` for automatic differentiation.")
            dU_single = jax.grad(U_single)
        else:
            print("Manual `dU` supplied by the user.")

        # Vectorize the functions to handle multiple walkers efficiently.
        self.U = jax.vmap(U_single, in_axes=1, out_axes=0)
        self.dU = jax.vmap(dU_single, in_axes=1, out_axes=1)

        self.key = random.PRNGKey(42)

    def _leapfrog(self, qi, pi, lf_length, steps):
        """
        A lean leapfrog integrator that does NOT store the orbit,
        optimized with jax.lax.fori_loop.
        """
        epsilon = lf_length / steps
        
        # Initial half-step for momentum
        p = pi - (epsilon / 2.0) * self.dU(qi)
        q = qi

        # Main loop using JAX's optimized looping construct
        def body_fun(i, state):
            q, p = state
            q_new = q + epsilon * p / self.m
            p_new = p - epsilon * self.dU(q_new)
            return (q_new, p_new)

        # The loop runs for steps-1 full steps
        q, p = jax.lax.fori_loop(0, steps - 1, body_fun, (q, p))

        # Final full step for position and half-step for momentum
        q = q + epsilon * p / self.m
        p = p - (epsilon / 2.0) * self.dU(q)
        
        return q, -p

    def _leapfrog_with_orbit(self, qi, pi, lf_length, steps):
        """
        Leapfrog integrator that stores the orbit, optimized with jax.lax.fori_loop.
        """
        epsilon = lf_length / steps
        n_walkers_device = qi.shape[1]
        n_parameters = qi.shape[0]
        
        # Initial half-step for momentum
        p = pi - (epsilon / 2.0) * self.dU(qi)
        q = qi

        # Initialize orbit storage
        q_orbit = jnp.zeros((steps, n_parameters, n_walkers_device))
        q_orbit = q_orbit.at[0].set(q)
        
        # Main loop using JAX's optimized looping construct
        def body_fun(i, state):
            q, p, q_orbit = state
            q_new = q + epsilon * p / self.m
            p_new = p - epsilon * self.dU(q_new)
            q_orbit = q_orbit.at[i + 1].set(q_new)
            return (q_new, p_new, q_orbit)

        # The loop runs for steps-1 full steps
        q, p, q_orbit = jax.lax.fori_loop(0, steps - 1, body_fun, (q, p, q_orbit))

        # Final full step for position and half-step for momentum
        q = q + epsilon * p / self.m
        p = p - (epsilon / 2.0) * self.dU(q)
        
        # The final position is q, not q_orbit[-1], to avoid an extra array lookup
        return q, -p, q_orbit


    def initialise_parameters(self):
        """Initializes parameters for the HMC run."""
        if self.state is None:
            self.n = self.n_samples + self.n_burnin
            self.n_collect = self.n
        else:
            self.n += self.n_samples
            self.n_collect = self.n_samples + 1

        self.initial_chains = self.chains
        self.initial_acceptance = self.acceptance

    def _step(self, q, key, lf_length, steps, store_orbits):
        """
        Generates a new sample using one HMC step. This function is executed
        in parallel on each device, receiving a slice of the walkers and
        the leapfrog parameters for this specific step.
        """
        key, p_key, accept_key = random.split(key, 3)
        
        n_walkers_device = q.shape[1]

        # 1. Give each walker a random momentum
        p = self.p0 * random.normal(p_key, shape=q.shape)

        # 2. Traverse phase space, passing parameters explicitly
        if store_orbits:
            qf_prop, pf_prop, orbit = self._leapfrog_with_orbit(q, p, lf_length, steps)
        else:
            qf_prop, pf_prop = self._leapfrog(q, p, lf_length, steps)
            # Create a dummy orbit if not storing to maintain function signature
            orbit = jnp.zeros((steps, self.n_parameters, n_walkers_device))


        # 3. Determine initial and final energies
        Ui = self.U(q)
        Ki = jnp.sum(p**2, axis=0) / 2.0
        Uf = self.U(qf_prop)
        Kf = jnp.sum(pf_prop**2, axis=0) / 2.0

        # 4. Determine change in energy
        deltaE = jnp.exp(Ui - Uf + Ki - Kf)

        # 5. Accept or reject the proposal
        acceptance_rand = random.uniform(accept_key, shape=(n_walkers_device,))
        accepted = acceptance_rand < deltaE
        
        qf_final = jnp.where(accepted[:, None], qf_prop.T, q.T).T

        return qf_final, orbit, accepted, key

    def run_hmc(self):
        """
        Executes the HMC sampling process in parallel across available devices.
        """
        if self.qi is None:
            raise ValueError("Attribute `self.qi` must be set before running the sampler.")

        num_devices = jax.device_count()
        if self.n_walkers % num_devices != 0:
            raise ValueError(f"Number of walkers ({self.n_walkers}) must be divisible by "
                             f"the number of devices ({num_devices}).")
        

        walkers_per_device = self.n_walkers // num_devices
        print(f"Deploying {self.n_walkers} walkers across {num_devices} devices ({walkers_per_device} per device).")

        self.n_parameters = self.qi.shape[0]
        self.initialise_parameters()
        
        # --- Create the pmapped function for this specific run ---
        pmapped_step = jax.pmap(
            functools.partial(self._step, lf_length=self.lf_length, steps=self.steps, store_orbits=self.store_orbits)
        )

        # --- Initialize Walkers ---
        q_sample = jnp.zeros((self.n_parameters, self.n_collect, self.n_walkers))
        if self.state is None:
            key, subkey = random.split(self.key)
            initial_positions = jnp.expand_dims(self.qi, 1) + 0.01 * random.normal(subkey, shape=(self.n_parameters, self.n_walkers))
            q_sample = q_sample.at[:, 0, :].set(initial_positions)
        else:
            q_sample = q_sample.at[:, 0, :].set(self.state)

        if self.store_orbits:
            q_orbit = jnp.zeros((self.n_parameters, self.steps, self.n_collect, self.n_walkers))
        
        q_acceptance = jnp.zeros((self.n_collect, self.n_walkers))
        
        # Split the key for each device
        device_keys = random.split(self.key, num_devices)

        with tqdm(total=self.n_collect - 1, desc="Collecting samples") as pbar:
            for ii in range(1, self.n_collect):
                # Get the current positions and reshape for pmap
                q_current_flat = q_sample[:, ii - 1, :]
                q_current_sharded = q_current_flat.reshape(self.n_parameters, num_devices, walkers_per_device).transpose(1, 0, 2)
                
                # Execute one parallel step
                q_next_sharded, orbit_sharded, accepted_sharded, device_keys = pmapped_step(
                    q=q_current_sharded, 
                    key=device_keys
                )
                
                # Reshape results back from sharded to flat
                q_next_flat = q_next_sharded.transpose(1, 0, 2).reshape(self.n_parameters, self.n_walkers)
                accepted_flat = accepted_sharded.flatten()

                # Update history
                q_sample = q_sample.at[:, ii, :].set(q_next_flat)
                q_acceptance = q_acceptance.at[ii, :].set(accepted_flat)
                
                if self.store_orbits:
                    orbit_flat = orbit_sharded.transpose(2, 1, 0, 3).reshape(self.n_parameters, self.steps, self.n_walkers)
                    q_orbit = q_orbit.at[:, :, ii - 1, :].set(orbit_flat)

                if self.save_every is not None and ii > 1 and ii % self.save_every == 0:
                    self.save_chains(q_sample[:, :ii, :])
                    self.save_samples()
                    self.save_acceptance(q_acceptance[:ii, :])
                    self.state = q_sample[:, ii, :]
                    self.save()

                pbar.update(1)
        
        # The final key is the set of keys from all devices. We just need one for continuation.
        self.key = device_keys[0]

        # --- Finalize and Save Results ---
        self.save_chains(q_sample)
        self.save_samples()
        self.save_acceptance(q_acceptance)
        if self.store_orbits:
            self.save_orbits(q_orbit)
        self.state = q_sample[:, -1, :]
        self.calculate_medians()
        self.calculate_covariance()
        return True

    def _integrated_autocorrelation_time(self, x):
        """
        Calculates the integrated autocorrelation time (IAT) of a time series.
        This implementation is based on the method described in the `emcee` documentation.
        """
        # Ensure input is a numpy array
        x = np.asarray(x)
        # Center the data
        x = x - np.mean(x)
        N = len(x)
        
        # Compute the autocorrelation function using FFT
        f = np.fft.fft(x, n=2*N)
        acf = np.fft.ifft(f * np.conj(f))[:N].real
        acf /= acf[0]
        
        # Automated windowing procedure to find where the ACF first becomes negative
        try:
            # Find the first lag where the ACF is negative
            M = np.where(acf < 0)[0][0]
        except IndexError:
            # If the ACF is always positive, use a fallback window size
            M = N // 2

        # The IAT is 1 + 2 * sum(ACF from lag 1 to M)
        tau = 1 + 2 * np.sum(acf[1:M])
        return tau

    def tune_leapfrog(self, qi_center, lf_lengths, steps_list, n_tune_samples=1000, n_tune_burnin=500):
        """
        Performs a grid search to find optimal leapfrog parameters.
        
        Parameters:
            key (jax.random.PRNGKey): JAX random key for the tuning run.
            qi_center (jnp.ndarray): The central point for the initial walker positions.
            lf_lengths (list): A list of leapfrog path lengths to test.
            steps_list (list): A list of leapfrog steps to test.
            n_tune_samples (int): Number of samples to draw for each tuning run.
            n_tune_burnin (int): Number of burn-in samples to discard for each tuning run.
        
        Returns:
            list: A list of dictionaries, each containing the results for one parameter combination.
        """
        print("\n--- Commencing Leapfrog Parameter Tuning ---")
        results = []
        
        self.n_parameters = qi_center.shape[0]

        for lf in lf_lengths:
            for steps in steps_list:
                # --- Setup for this grid point ---
                epsilon = lf / steps
                
                print(f"\nTuning with lf_length={lf}, steps={steps} (epsilon={epsilon:.4f})")
                start_time = time.time()

                # --- Create a pmapped function specifically for this tuning run ---
                # Tuning runs never need to store orbits, so we set store_orbits=False
                pmapped_step = jax.pmap(
                    functools.partial(self._step, lf_length=lf, steps=steps, store_orbits=False)
                )

                # --- Run a self-contained HMC campaign ---
                num_devices = jax.device_count()
                walkers_per_device = self.n_walkers // num_devices
                
                self.key, init_key, run_key = random.split(self.key, 3)
                device_keys = random.split(run_key, num_devices)
                
                initial_positions = jnp.expand_dims(qi_center, 1) + 0.01 * random.normal(init_key, shape=(self.n_parameters, self.n_walkers))

                local_chains = jnp.zeros((self.n_parameters, n_tune_samples, self.n_walkers))
                local_chains = local_chains.at[:, 0, :].set(initial_positions)
                local_acceptance = jnp.zeros((n_tune_samples, self.n_walkers))
                
                # Main tuning loop
                for ii in tqdm(range(1, n_tune_samples), desc="Tuning run", leave=False):
                    q_current_flat = local_chains[:, ii - 1, :]
                    q_current_sharded = q_current_flat.reshape(self.n_parameters, num_devices, walkers_per_device).transpose(1, 0, 2)
                    
                    # Pass parameters explicitly to the pmapped function
                    q_next_sharded, _, accepted_sharded, device_keys = pmapped_step(
                        q=q_current_sharded, 
                        key=device_keys
                    )
                    
                    q_next_flat = q_next_sharded.transpose(1, 0, 2).reshape(self.n_parameters, self.n_walkers)
                    accepted_flat = accepted_sharded.flatten()

                    local_chains = local_chains.at[:, ii, :].set(q_next_flat)
                    local_acceptance = local_acceptance.at[ii, :].set(accepted_flat)
                
                runtime = time.time() - start_time

                # --- Analyze results ---
                chains_after_burn = local_chains[:, n_tune_burnin:, :]
                acceptance_after_burn = local_acceptance[n_tune_burnin:, :]
                
                samples_for_iat = chains_after_burn.transpose((2, 1, 0)).reshape(-1, self.n_parameters)
                
                mean_acceptance = jnp.mean(acceptance_after_burn)
                iat_per_param = [self._integrated_autocorrelation_time(samples_for_iat[:, i]) for i in range(self.n_parameters)]
                mean_iat = np.mean(iat_per_param)
                
                results.append({
                    'lf_length': lf,
                    'steps': steps,
                    'epsilon': epsilon,
                    'acceptance_rate': float(mean_acceptance),
                    'mean_iat': float(mean_iat),
                    'runtime': runtime
                })
        
        print("\n--- Leapfrog Tuning Results ---")
        print("-" * 80)
        print(f"{'lf_length':<12} | {'steps':<8} | {'epsilon':<10} | {'acceptance':<12} | {'mean_iat':<12} | {'runtime (s)':<12}")
        print("-" * 80)
        for res in results:
            print(f"{res['lf_length']:<12.3f} | {res['steps']:<8} | {res['epsilon']:<10.4f} | {res['acceptance_rate']:<12.3f} | {res['mean_iat']:<12.2f} | {res['runtime']:<12.2f}")
        print("-" * 80)

        # Find the best parameters based on a simple heuristic:
        # Choose the run with an acceptance rate between 0.6 and 0.8 that has the lowest IAT.
        best_params = None
        min_iat = float('inf')
        for res in results:
            if 0.6 < res['acceptance_rate'] < 0.95:
                if res['mean_iat'] < min_iat:
                    min_iat = res['mean_iat']
                    best_params = res

        if best_params:
            print(f"\nOptimal parameters found: lf_length={best_params['lf_length']}, steps={best_params['steps']}")
            self.lf_length = best_params['lf_length']
            self.steps = best_params['steps']
        else:
            print("\nNo parameters found in the ideal acceptance range. Using defaults.")
        
        return results

    # --- Utility and Plotting Functions ---

    def save_chains(self, q_sample):
        """Saves chains. Handles concatenation if continuing a run."""
        if self.initial_chains is None:
            self.chains = q_sample
        else:
            self.chains = jnp.concatenate((self.initial_chains, q_sample[:, 1:, :]), axis=1)

    def save_acceptance(self, q_acceptance):
        """Saves acceptance rates."""
        if self.initial_acceptance is None:
            self.acceptance = q_acceptance
        else:
            self.acceptance = jnp.concatenate((self.initial_acceptance, q_acceptance[1:, :]), axis=0)

    def save_samples(self):
        """Extracts and saves post-burn-in samples."""
        if self.chains.shape[1] > self.n_burnin:
            samples_raw = self.chains[:, self.n_burnin:, :]
            self.samples = samples_raw.transpose((2, 1, 0)).reshape(-1, self.n_parameters)
        else:
            self.samples = None
            
    def save_orbits(self, q_orbit):
        """Saves orbits."""
        if self.chains.shape[1] > self.n_burnin:
            orbits_raw = q_orbit[:, :, self.n_burnin:, :]
            self.orbits = orbits_raw.transpose((2,3,1,0)).reshape(-1, self.steps, self.n_parameters)
        else:
            self.orbits = None

    def plot_samples(self, labels: list = None, truths: list = None):
        """Produces a corner plot of the samples."""
        if self.samples is None:
            print("No samples to plot. Run the sampler first.")
            return None
        if labels is None and self.dim_labels is not None:
            labels = self.dim_labels
        
        samples_np = np.asarray(self.samples)
        
        figure = corner.corner(
            samples_np,
            labels=labels,
            truths=truths,
            quantiles=[0.16, 0.5, 0.84],
            show_titles=True,
            title_kwargs={"fontsize": 12}
        )
        return figure
    
    def plot_chains(self, labels: list = None):
        """Plots the walker chains over iterations."""
        if self.chains is None:
            print("No chains to plot.")
            return
        if labels is None and self.dim_labels is not None:
            labels = self.dim_labels
        
        chains_np = np.asarray(self.chains)
        samples_np = np.asarray(self.samples) if self.samples is not None else None
        
        figure, ax = plt.subplots(self.n_parameters + 1, sharex='col', figsize=(8, 1.5 * (self.n_parameters + 1)))
        
        for i in range(self.n_parameters):
            if samples_np is not None:
                p16, p50, p84 = np.percentile(samples_np[:, i], [16, 50, 84])
                ax[i].axhline(p16, color='r', linestyle='-.', lw=1)
                ax[i].axhline(p50, color='r', linestyle='--', lw=1)
                ax[i].axhline(p84, color='r', linestyle='-.', lw=1)

            ax[i].plot(chains_np[i, :, :], 'k', alpha=0.1)
            yl = ax[i].get_ylim()
            ax[i].axvline(self.n_burnin, color='b', linestyle='--')
            ax[i].set_ylabel(labels[i] if labels else f'Param {i}')
            ax[i].set_xlim([-1, self.n + 1])
            ax[i].set_ylim(yl)

        n_params, n_collect, n_walkers = chains_np.shape
        chains_reshaped = chains_np.reshape((n_params, n_collect * n_walkers))
        U_vals_flat = self.U(chains_reshaped)
        U_vals = np.asarray(U_vals_flat).reshape((n_collect, n_walkers))

        ax[self.n_parameters].plot(U_vals, 'k', alpha=0.1)
        ax[self.n_parameters].set_ylabel('U')
        ax[self.n_parameters].set_xlim([-1, self.n + 1])
        ax[self.n_parameters].set_xlabel('Iteration')
        
        plt.tight_layout(pad=0.4, w_pad=0.1, h_pad=0.1)
        return figure
    
    def _plot_ellipse(self, ax, mean, cov, n_std=1.0, **kwargs):
        """Helper function to plot a 2D covariance ellipse."""
        # Get eigenvalues and eigenvectors
        vals, vecs = np.linalg.eigh(cov)
        # Get angle of rotation
        x, y = vecs[:, 0]
        angle = np.degrees(np.arctan2(y, x))

        # Get ellipse width and height
        # The chi2 distribution is used to find the scaling factor for the desired confidence level
        # For 1-sigma in 2D, this is sqrt(2.30)
        # For 2-sigma in 2D, this is sqrt(6.18)
        s = np.sqrt(2.30) if n_std == 1.0 else np.sqrt(6.18) if n_std == 2.0 else n_std
        width, height = 2 * s * np.sqrt(vals)
        
        # Create ellipse
        ellipse = Ellipse(xy=mean, width=width, height=height, angle=angle, **kwargs)
        ax.add_patch(ellipse)

    def plot_covariance_comparison(self, means, cov_dict, labels=None, truths=None, n_std=1.0):
        """
        Creates a corner plot comparing multiple covariance matrices as ellipses.

        Parameters:
            means (np.ndarray): The mean vector for the parameters, shape (n_parameters,).
            cov_dict (dict): A dictionary where keys are method names (str) and
                             values are covariance matrices (np.ndarray).
            labels (list, optional): Names of the parameters for plot axes.
            truths (list, optional): True values of parameters to overplot.
            n_std (float, optional): The number of standard deviations for the ellipse contour (1.0 or 2.0).
        """
        n_dim = len(means)
        if n_dim < 2:
            print("Covariance comparison plot requires at least 2 dimensions.")
            return

        if labels is None:
            labels = [f'p{i}' for i in range(n_dim)]

        fig, axes = plt.subplots(n_dim - 1, n_dim - 1, figsize=(3 * (n_dim - 1), 3 * (n_dim - 1)))
        # Ensure axes is a 2D array even for n_dim=2
        if n_dim == 2:
            axes = np.array([[axes]])

        colors = plt.cm.viridis(np.linspace(0, 1, len(cov_dict)))

        for i in range(1, n_dim):
            for j in range(i):
                # Map from parameter indices (i, j) to subplot grid (row, col)
                row, col = i - 1, j
                ax = axes[row, col]

                # --- Off-diagonal plots (2D ellipses) ---
                for k, (name, cov) in enumerate(cov_dict.items()):
                    # Extract the 2x2 sub-matrix for parameters j and i
                    cov_2d = cov[np.ix_([j, i], [j, i])]
                    mean_2d = means[[j, i]]
                    self._plot_ellipse(ax, mean_2d, cov_2d, n_std=n_std, 
                                       facecolor='none', edgecolor=colors[k], lw=1.5, label=name)
                if truths is not None:
                    ax.plot(truths[j], truths[i], 'rs', markersize=5)
                
                # --- Set plot limits based on the largest covariance ---
                all_covs = [c[np.ix_([j, i], [j, i])] for c in cov_dict.values()]
                max_std = np.max([np.sqrt(np.diag(c)) for c in all_covs], axis=0) * (n_std * 2.5) # Increased buffer
                ax.set_xlim(means[j] - max_std[0], means[j] + max_std[0])
                ax.set_ylim(means[i] - max_std[1], means[i] + max_std[1])

                # --- Labels and Ticks ---
                if col == 0:
                    ax.set_ylabel(labels[i])
                if row == n_dim - 2:
                    ax.set_xlabel(labels[j])
                
                if row < n_dim - 2:
                    ax.set_xticklabels([])
                if col > 0:
                    ax.set_yticklabels([])
        
        # Turn off unused upper-triangle axes
        for i in range(n_dim - 1):
            for j in range(n_dim - 1):
                if j > i:
                    axes[i, j].axis('off')

        # Create a single legend for the whole figure
        handles = [plt.Line2D([0], [0], color=c, lw=2) for c in colors]
        fig.legend(handles, cov_dict.keys(), loc='upper right')
        
        fig.tight_layout(pad=0.5)
        plt.show()

    def reset(self):
        """Resets all simulation attributes."""
        self.state = None
        self.chains = None
        self.samples = None
        self.orbits = None
        self.acceptance = None
        self.qi = None

    def save(self, file_name: str):
        """
        Saves the serializable state of the sampler to a file using pickle.
        This method automatically saves any attribute that is not a function.
        """
        state_dict = {}
        for key, value in self.__dict__.items():
            # Exclude callables (functions, methods) which are not serializable
            if not callable(value):
                state_dict[key] = value

        with open(file_name, "wb") as file:
            pickle.dump(state_dict, file)
        print(f"Sampler state saved to {file_name}")

    def load(self, file_name: str):
        """
        Loads the state of a sampler from a file.
        The sampler object must be instantiated with the correct U function
        (or model/data for subclasses) before calling this method.
        """
        with open(file_name, "rb") as file:
            state_dict = pickle.load(file)
        
        for key, value in state_dict.items():
            setattr(self, key, value)
        
        print(f"Sampler state loaded from {file_name}")

    def calculate_medians(self):
        if self.samples is not None:
            self.q_medians = np.median(self.samples, axis=0)

    def calculate_covariance(self):
        if self.samples is not None:
            self.q_covariance = np.cov(self.samples.T)


# --- (NEW) Specialized Subclass for Bayesian Inference ---
class BayesianHMCSampler(HMCSampler):
    """
    A specialized HMCSampler for Bayesian inference.

    This class constructs the potential energy function from a log-likelihood and
    a log-prior, simplifying the user workflow for Bayesian problems.
    """
    def __init__(self, log_likelihood, log_prior=None, prior_mean=None, prior_sd=None, dU=None):
        """
        Initializes the Bayesian sampler.

        Parameters:
            log_likelihood (callable): The log-likelihood function, `log(P(data|theta))`.
            log_prior (callable, optional): The log-prior function, `log(P(theta))`.
                                            It must have the signature `log_prior(theta, mean, sd)`.
                                            Takes precedence over the default Gaussian prior.
            prior_mean (jnp.ndarray, optional): Mean for a Gaussian prior.
            prior_sd (jnp.ndarray, optional): Standard deviation for a Gaussian prior.
            dU (callable, optional): A manual gradient function for the potential energy.
        """
        # --- Define default prior functions with the new, flexible signature ---
        def gaussian_log_prior(theta, mean, sd):
            if mean is None or sd is None:
                return 0.0 # Behave like a uniform prior if params are missing
            return -0.5 * jnp.sum(((theta - mean) / sd)**2)

        def uniform_log_prior(theta, mean, sd):
            return 0.0

        # --- Select the prior function to use ---
        if log_prior is not None:
            # User provided a custom prior function
            selected_log_prior = log_prior
        elif prior_mean is not None and prior_sd is not None:
            # User provided parameters for the default Gaussian prior
            selected_log_prior = gaussian_log_prior
        else:
            # Default to a uniform prior
            selected_log_prior = uniform_log_prior

        # --- Create the final potential energy function ---
        # It combines the likelihood and the selected prior.
        def potential_energy(theta):
            # The selected log_prior function is called with the stored mean and sd
            log_prior_val = selected_log_prior(theta, prior_mean, prior_sd)
            log_likelihood_val = log_likelihood(theta)
            return -(log_likelihood_val + log_prior_val)
        
        super().__init__(U=potential_energy, dU=dU)

    # Alias for run_hmc
    sample_posterior = HMCSampler.run_hmc


# --- Specialized Subclass for Model Fitting, inheriting from Bayesian Sampler ---
class ModelFitterHMC(BayesianHMCSampler):
    """
    A specialized HMCSampler for fitting a model to data, assuming a Gaussian likelihood.
    """
    def __init__(self, model, x_data, y_data, y_err=1.0, 
                 log_prior=None, prior_mean=None, prior_sd=None, dU=None):
        """
        Initializes the model-fitting sampler.

        Parameters:
            model (callable): The model function, `model(theta, x)`.
            x_data, y_data, y_err: The data and uncertainties.
            log_prior, prior_mean, prior_sd: Options for the prior, passed to BayesianHMCSampler.
            dU (callable, optional): A manual gradient function.
        """
        self.model = model
        self.x_data = x_data
        self.y_data = y_data
        self.y_err = y_err

        # Define the log-likelihood based on a Gaussian assumption (chi-squared)
        def log_likelihood(theta):
            model_y = self.model(theta, self.x_data)
            chi_squared = jnp.sum(((self.y_data - model_y) / self.y_err)**2)
            return -chi_squared / 2.0
        
        # Call the parent BayesianHMCSampler's __init__
        super().__init__(log_likelihood=log_likelihood, log_prior=log_prior, 
                         prior_mean=prior_mean, prior_sd=prior_sd, dU=dU)

    # Alias for run_hmc
    fit_model = HMCSampler.run_hmc



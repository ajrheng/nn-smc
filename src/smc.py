import numpy as np
from .functions import (
    prob_zero,
    prob_one,
    weighted_std,
)
from .getdatahelper import (
    make_batch
)

class phase_est_smc:

    def __init__(self, true_omega, t0, max_iters):
        self.true_omega = true_omega
        self.t = t0
        self.break_flag = False
        self.counter = 0
        self.data = []
        self.std_list = []
        self.max_iters = max_iters
        self.curr_omega_est = 0 # best current estimate of omega
        self.rng = np.random.default_rng()

        self.errors_list = []
        self.times_list = []

    def init_particles(self, num_particles):
        """
        Initializes the particles for SMC.

        Args:
            num_particles: number of particles in the SMC algorithm
        """
        self.num_particles = num_particles
        self.particle_pos = self.rng.uniform(-np.pi, np.pi, size=self.num_particles)
        self.particle_wgts = np.ones(num_particles) * 1/num_particles # uniform weight initialization

    def particles(self, threshold=None):
        """
        Runs the SMC algorithm for current experiment until the threshold exceeded

        Args:
            n_measurements: number of measurements per update
            threshold: threshold for n_eff. defaults to self.num_particles/10

        Returns:
            array of particle positions and their corresponding weights
        """
        
        if threshold is None:
            threshold = self.num_particles/10

        n_eff = None # init None so it will be calculated on first iteration of while loop

        while n_eff is None or n_eff >= threshold:

            phi = self.rng.uniform(low=-1, high=1) * np.pi

            r = self.rng.uniform()
            if r <= prob_zero(self.true_omega, phi, self.t):
                likelihood = prob_zero(self.particle_pos, phi, self.t)
            else:
                likelihood = prob_one(self.particle_pos, phi, self.t)

            # bayes update of weights
            self.particle_wgts = np.multiply(self.particle_wgts, likelihood) # numerator
            norm = np.sum(self.particle_wgts) # denominator
            self.particle_wgts /= norm
            
            # recalculate n_eff
            n_eff = 1/(np.sum(self.particle_wgts**2))
            self.data.append(np.average(self.particle_pos, weights = self.particle_wgts))

            # store standard deviation of run
            std = weighted_std(self.particle_pos, self.particle_wgts)
            self.std_list.append(std)
          
            # self.curr_omega_est = self.particle_pos[np.argmax(self.particle_wgts)]
            avg = np.average(self.particle_pos, weights = self.particle_wgts)
            self.curr_omega_est = avg

            self.errors_list.append((self.curr_omega_est-self.true_omega)**2)
            self.times_list.append(self.t)

            # exit condition
            self.counter += 1
            if self.counter == self.max_iters:
                self.break_flag=True
                break

            self.update_t()

        return self.particle_pos, self.particle_wgts

    def update_t(self):
        """
        Updates time 
        """

        self.t = self.t * 9/8

    def bootstrap_resample(self):
        """
        Simple bootstrap resampler
        """
        
        self.particle_pos = self.rng.choice(self.particle_pos, size = self.num_particles, p=self.particle_wgts)
        self.particle_wgts = np.ones(self.num_particles) * 1/self.num_particles

    def liu_west_resample(self, a=0.98):
        """
        Liu-West resampler
        """
        
        # e_x2 = np.average(self.particle_pos**2, weights=self.particle_wgts)
        mu = np.average(self.particle_pos, weights=self.particle_wgts)
        var = weighted_std(self.particle_pos, self.particle_wgts) ** 2
        var = (1-a**2) * var
        new_particle_pos = self.rng.choice(self.particle_pos, size=self.num_particles, p=self.particle_wgts)
        for i in range(len(new_particle_pos)):
            mu_i = a * new_particle_pos[i] + (1-a) * mu
            new_particle_pos[i] = self.rng.normal(loc=mu_i, scale=np.sqrt(var))  ## scale is standard deviation

        self.particle_pos = np.copy(new_particle_pos)
        self.particle_wgts = np.ones(self.num_particles) * 1/self.num_particles ## set all weights to 1/N again
 
    def sample_from_posterior(self, n_batch=100, n_samples=10000, n_bins=50):

        """
        n_batch: number of points in a batch
        n_samples: number of samples to draw from posterior for binning
        n_bins: number of bins to bin the samples to (resolution of data)
        """

        particle_pos = self.particle_pos.copy()
        sampled_particles = self.rng.choice(particle_pos, size=n_samples, p=self.particle_wgts)
        bins, edges = np.histogram(sampled_particles, bins=n_bins)
        bins = bins/n_samples
        original_bins = bins.copy()

        # batch: [n_batch, n_bins*n_particles_per_sample/downsample_factor]
        batch = make_batch(bins, n_batch, n_particles_per_sample=10)
        return batch, original_bins, edges


    def convert_to_particles(self, batch, edge, mean=None, std=None):

        # batch: eg. [500 x 50] means 500 particles of resolution 50 bins
        # edge: eg. [51] representing the left and right edges of 50 bins

        assert batch.shape[0] == self.num_particles
        
        n_bins = batch.shape[-1]
        bin_integers = np.arange(n_bins, dtype=int)
        particle_pos = []

        # redefine edges in case NN output is higher resolution than input
        edge = np.linspace(edge[0], edge[-1], num=n_bins+1, endpoint=True)

        if mean is not None and std is not None:
            edge = (edge * std) + mean

        for i in range(len(batch)):
            probabilities = batch[i]
            particle_integer = self.rng.choice(bin_integers, p=probabilities)
            particle_pos_ = self.rng.uniform(edge[particle_integer], edge[particle_integer+1])
            particle_pos.append(particle_pos_)
        
        self.particle_pos = np.array(particle_pos)
        self.particle_wgts = np.ones_like(self.particle_pos) / self.num_particles

    def credible_region(self,  level=0.95):

        id_sort = np.argsort(self.particle_wgts)[::-1]

        cumsum_weights = np.cumsum(self.particle_wgts[id_sort])

        id_cred = cumsum_weights <= level
        id_cred[np.sum(id_cred)] = True

        return self.particle_pos[id_sort][id_cred]


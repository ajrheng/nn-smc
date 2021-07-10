import yaml
import numpy as np
from src.smc import phase_est_smc
import os
from tqdm import tqdm


if __name__ == "__main__":

    with open("get_data_config.yaml", 'r') as file:
        config = yaml.load(file, Loader=yaml.FullLoader)

    n_particles = config['n_particles']
    n_data = config['n_data']
    n_bins = config['n_bins']
    max_iters = config['max_iters']
    n_batch = config['n_batch']
    t0 = config['t0']
    directory = config['directory']
    data_filename = config['data_filename']

    batch_data = []
    bin_data = []
    edge_data = []

    for _ in tqdm(range(n_data)):

        true_omega = np.random.uniform(low=-np.pi, high =np.pi)
        smc = phase_est_smc(true_omega, t0, max_iters)
        smc.init_particles(n_particles)

        while True:

            smc.particles(threshold=n_particles/30)
            if smc.break_flag == True:
                break
            else:
                batch, bins, edges = smc.sample_from_posterior(n_batch=n_batch, n_bins=n_bins)
                batch_data.append(batch)
                bin_data.append(bins)
                edge_data.append(edges)

            smc.bootstrap_resample()

    batch_data = np.array(batch_data)
    bin_data = np.array(bin_data)
    edge_data = np.array(edge_data)

    if not os.path.exists(directory):
        os.makedirs(directory)
    filepath = os.path.join(directory, data_filename)

    np.savez_compressed(filepath,
                        batch_data=batch_data, 
                        bin_data=bin_data, 
                        edge_data=edge_data
                        )
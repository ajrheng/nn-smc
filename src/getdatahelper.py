import numpy as np


def unary_particles_from_bin(bins, n_particles=10):
    
    n_bins = len(bins)
    positions = np.arange(n_bins, dtype=int)
    particle_positions = np.random.choice(positions, n_particles, p=bins)

    data = []
    for x in particle_positions:
        unary_encoding = int_to_unary(x, n_bins)
        data.extend(unary_encoding)
    
    return np.array(data)


def int_to_unary(x, n_bins):
    
    assert n_bins > x, "n_bins greater than integer"
    
    unary_encoding = np.zeros(n_bins, dtype=int)
    unary_encoding[x] = 1
    return unary_encoding


def make_batch(bins, n_batch, n_particles_per_sample=10):
    batch_data = []
    for i in range(n_batch):
        unary = unary_particles_from_bin(bins, n_particles_per_sample)
        batch_data.append(unary)

    return np.array(batch_data)
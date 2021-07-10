import torch
import math
import torch.nn as nn
import torch.nn.functional as F
import numpy as np
import torch.optim as optim
import matplotlib.pyplot as plt
from tqdm import tqdm
import yaml
import os
import seaborn as sns
sns.set()

from src.mlp import model
from src.smc import phase_est_smc


if __name__ == "__main__":

    with open("./config/run_config.yaml", 'r') as file:
        config = yaml.load(file, Loader=yaml.FullLoader)

    n_particles = config['n_particles']
    n_runs = config['n_runs']
    t0 = config['t0']
    max_iters = config['max_iters']
    directory = config['directory']

    net = model()
    net.load_state_dict(torch.load("./files/model.pt"))
    net.eval()

    if not os.path.exists(directory):
        os.makedirs(directory)
    log_path = os.path.join(directory, "log.txt")

    true_omegas = []
    nn_preds = []
    nn_data = []
    cred_reg_data = []
    restart_list = []
    crb_list = []

    for i in range(n_runs): 
        
        true_omega = np.random.uniform(low=-1, high =1) * np.pi
        true_omegas.append(true_omega)
        restart_counts = 0
        
        while True:
        
            smc = phase_est_smc(true_omega, t0, max_iters)
            smc.init_particles(n_particles)
            resample_counts = 0
            
            while True:

                smc.particles(threshold=n_particles/30)

                if smc.break_flag is True:
                    break
                else:
                    batch, _, edges = smc.sample_from_posterior(n_batch = n_particles)
                    batch = torch.from_numpy(batch).float()
                    predictions = net(batch).detach().numpy()
                    smc.convert_to_particles(predictions, edges)
                    resample_counts += 1

            final_std = smc.std_list[-1]
            if final_std < 1e-7:
                break

            restart_counts += 1
            
        nn_data.append(smc.data)
        nn_preds.append(smc.curr_omega_est)
        cred_reg_data.append(smc.cred_reg_list)
        restart_list.append(restart_counts)
        crb_list.append(smc.crb_list)
        
        with open(log_path, 'a') as out_file:
            out_file.write("True omega: {:f}, pred: {:f}, n_resample: {:d}, n_restarts: {:d}".format(true_omega,
                                                                                    smc.curr_omega_est,
                                                                                    resample_counts,
                                                                                    restart_counts))
            out_file.write("Final standard deviation: {}".format(final_std))


import torch
from datetime import datetime
import torch.nn as nn
import torch.nn.functional as F
import numpy as np
import torch.optim as optim
import matplotlib.pyplot as plt
import yaml
import os
import argparse
import seaborn as sns
sns.set()

from src.nn import neural_network
from src.smc import phase_est_smc


if __name__ == "__main__":

    config_file = "pass_fail.yaml"
    config_path = os.path.join("./config", config_file)

    with open(config_path, 'r') as file:
        config = yaml.load(file, Loader=yaml.FullLoader)

    n_particles = config['n_particles']
    n_runs = config['n_runs']
    t0 = config['t0']
    max_iters = config['max_iters']
    directory = config['directory']
    n_eff_thresh = config['n_eff_thresh']
    if isinstance(n_eff_thresh, str):
        n_eff_thresh = float(n_eff_thresh)
    
    time = datetime.now().strftime("%Y_%m_%d_%H%M%S")
    run_path = os.path.join(directory, 'nn', time)
    if not os.path.exists(run_path):
        os.makedirs(run_path)
    log_path = os.path.join(run_path, "log.txt")

    architecture_path = os.path.join(run_path, "architecture.txt")
    architecture =  '_'.join(str(e) for e in config['model_neurons'])
    with open(architecture_path, 'a') as out_file:
        out_file.write("Neural network architecture\n")
        out_file.write(architecture )

    units = config['model_neurons']
    model_path = './files/' + config['model_filename']
    net = neural_network(units)
    net.load_state_dict(torch.load(model_path))
    net.eval()

    passed_std = []
    failed_std = []
    true_omega = np.random.uniform(low=-1, high =1) * np.pi

    while len(passed_std) < n_runs or len(failed_std) < n_runs:

        smc = phase_est_smc(true_omega, t0, max_iters)
        smc.init_particles(n_particles)

        while True:

            smc.particles(threshold=n_particles/30)

            if smc.break_flag is True:
                break
            else:
                batch, _, edges = smc.sample_from_posterior(n_batch = n_particles)
                batch = torch.from_numpy(batch).float()
                predictions = net(batch).detach().numpy()
                smc.convert_to_particles(predictions, edges)

        final_std = smc.std_list[-1]
        if final_std <  n_eff_thresh and len(passed_std) < n_runs:
            passed_std.append(smc.std_list)
        elif len(failed_std) < n_runs:
            failed_std.append(smc.std_list)

        with open(log_path, 'a') as out_file:
            out_file.write("Length of passed: {}, failed: {} \n".format(len(passed_std), len(failed_std)))
        

    passed_std = np.array(passed_std).mean(axis=0)
    failed_std = np.array(failed_std).mean(axis=0)

    n_iters_arr = np.arange(max_iters, dtype=int)

    fig_path = "./figures/passed_failed.pdf"
    f, (ax1, ax2) = plt.subplots(1,2, figsize=(10,5))

    ax1.plot(n_iters_arr, passed_std)
    ax2.plot(n_iters_arr, failed_std, color='red')
    ax1.set_ylabel('$\sigma$', fontsize=13)
    f.text(0.5, 0.02, 'Iterations', ha='center', fontsize=13)
    f.text(0.1, 0.9, 'a)', fontsize=13)
    f.text(0.51, 0.9, 'b)', fontsize=13)
    ax1.set_yscale('log')
    ax1.tick_params(axis='both', which='major', labelsize=12)
    ax2.tick_params(axis='both', which='major', labelsize=12)
    f.savefig(fig_path,dpi=300)

    result_path = './files/results_pass_fail'

    np.savez_compressed(result_path,
                        passed_std = passed_std,
                        failed_std = failed_std
                        )
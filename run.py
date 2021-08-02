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

    parser = argparse.ArgumentParser()
    parser.add_argument(
        '--resampler',
        choices = ['nn', 'lw', 'bs'],
        type=str,
        help='Type of resampler. nn or lw or bs'
    )
    args, _ = parser.parse_known_args()

    config_file = "run_config_" + args.resampler + ".yaml"
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
    run_path = os.path.join(directory, args.resampler, time)
    if not os.path.exists(run_path):
        os.makedirs(run_path)
    log_path = os.path.join(run_path, "log.txt")

    if args.resampler == 'nn':

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
                    if args.resampler == 'nn':
                        batch, _, edges = smc.sample_from_posterior(n_batch = n_particles)
                        batch = torch.from_numpy(batch).float()
                        predictions = net(batch).detach().numpy()
                        smc.convert_to_particles(predictions, edges)
                    elif args.resampler == 'lw':
                        smc.liu_west_resample()
                    elif args.resampler == 'bs':
                        smc.bootstrap_resample()
                    resample_counts += 1

            final_std = smc.std_list[-1]
            if final_std < n_eff_thresh:
                break

            restart_counts += 1
            
        nn_data.append(smc.data)
        nn_preds.append(smc.curr_omega_est)
        cred_reg_data.append(smc.cred_reg_list)
        restart_list.append(restart_counts)
        crb_list.append(smc.crb_list)
        
        with open(log_path, 'a') as out_file:
            out_file.write("Run {:d} \n".format(i))
            out_file.write("True omega: {:f}, pred: {:f}, n_resample: {:d}, n_restarts: {:d} \n".format(true_omega,
                                                                                    smc.curr_omega_est,
                                                                                    resample_counts,
                                                                                    restart_counts))
            out_file.write("Final standard deviation: {}\n".format(final_std))
            out_file.write("-------------------------------\n\n")
        


    avg_restarts = np.mean(restart_list)
    nn_data_sq = ( np.array(nn_data) - np.array(true_omegas).reshape(-1,1)) ** 2
    nn_data_mean = np.mean(nn_data_sq, axis=0)
    nn_data_median = np.median(nn_data_sq, axis =0)

    cred_reg_data = np.array(cred_reg_data).mean(axis=0)
    cred_reg_min = cred_reg_data[:, 0]
    cred_reg_max = cred_reg_data[:, 1]

    results_path = os.path.join(run_path, "results.txt")
    with open(results_path, 'a') as out_file:
        out_file.write("Restart counts {}\n".format(avg_restarts))
        out_file.write("MSE: {}, Median: {}\n".format(nn_data_mean[-1], nn_data_median[-1]))

    n_iters_arr = np.arange(max_iters, dtype=int)

    f = plt.figure()
    plt.plot(n_iters_arr, nn_data_mean, label='Mean')
    plt.plot(n_iters_arr, nn_data_median, label='Median')
    plt.fill_between(n_iters_arr, cred_reg_max, cred_reg_min, alpha=0.4)
    plt.legend()
    plt.xlabel("Iterations")
    plt.ylabel("$(\omega - \omega*)^2$")
    plt.yscale('log')
    plt.xlim([0, max_iters])
    plt.tight_layout()
    fig_path = os.path.join(run_path, "smc_results.png")
    f.savefig(fig_path, dpi=300)

    if args.resampler == 'nn':
        result_path = './files/' + 'results_' + architecture
    elif args.resampler == 'bs':
        result_path = './files/results_bs'
    elif args.resampler == 'lw':
        result_path = './files/results_lw'

    np.savez_compressed(result_path,
                        mean_se=nn_data_mean,
                        median_se = nn_data_median,
                        cred_reg_max = cred_reg_max,
                        cred_reg_min = cred_reg_min
                        )
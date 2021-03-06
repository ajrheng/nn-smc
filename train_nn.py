import numpy as np
import torch
import torch.optim as optim
import torch.nn as nn
from sklearn.model_selection import train_test_split
from src.nn import neural_network
import sys
import matplotlib.pyplot as plt
from src.nn import model
import yaml
import os
from datetime import datetime
import seaborn as sns
import matplotlib.pyplot as plt
sns.set()


def train_step(tr_batch, tr_bin):
    
    # tr_batch = [n_batch, 500]
    tr_batch = tr_batch.to(device)
    tr_bin = tr_bin.to(device)
    train_pred = net(tr_batch)
    train_pred = train_pred.mean(dim=0) + epsilon
    train_pred = train_pred/torch.sum(train_pred)
    if (train_pred == 0).sum() > 0:
        print("ZEROES IN TRAIN")
        sys.exit()
    loss = loss_func(torch.log(train_pred), tr_bin)
    train_loss = loss.item()
    optimizer.zero_grad()
    loss.backward()
    optimizer.step()
    
    return train_loss

def test_step(test_batch, test_bin):
    
    net.eval()
    test_loss = 0
    test_len = len(test_batch)
    
    for i in range(test_len):
        te_batch = test_batch[i].to(device)
        te_bin = test_bin[i].to(device)
        test_pred = net(te_batch)
        test_pred = test_pred.mean(dim=0) + epsilon
        test_pred = test_pred/torch.sum(test_pred)
        if (test_pred == 0).sum() > 0:
            print("ZEROES IN TEST")
            sys.exit()
        loss = loss_func(torch.log(test_pred), te_bin)
        test_loss += loss.item()
        
    net.train()
    
    return test_loss/test_len

def load_data(data_filename):
    
    data_dict = np.load(data_filename)
    batch_data = data_dict["batch_data"]
    bin_data = data_dict["bin_data"]
    edge_data = data_dict["edge_data"]
    
    batch_data = torch.from_numpy(batch_data).float()
    bin_data = torch.from_numpy(bin_data).float()
    edge_data = torch.from_numpy(edge_data).float()
    
    train_batch, test_batch, train_bin, test_bin, train_edge, test_edge = \
        train_test_split(batch_data, bin_data, edge_data, test_size=0.2)
    train_dataset = torch.utils.data.TensorDataset(train_batch, train_bin, train_edge)
    train_data_loader = torch.utils.data.DataLoader(dataset=train_dataset, batch_size=1, shuffle=True)
    
    return train_data_loader, test_batch, test_bin, test_edge

if __name__ == "__main__":
    
    with open("./config/train_config.yaml", 'r') as file:
        config = yaml.load(file, Loader=yaml.FullLoader)

    device = torch.device('cpu')
    units = config['model_neurons']
    net = neural_network(units)
    net.train()

    lr = config['learning_rate']
    if isinstance(lr, str):
        lr = float(lr)

    optimizer = optim.Adam(net.parameters(), lr=lr)
    scheduler = optim.lr_scheduler.ReduceLROnPlateau(optimizer,
                                                     'min', 
                                                     factor=0.5, 
                                                     patience=3,
                                                     threshold=1e-5,
                                                     threshold_mode='abs'
                                                     )
    loss_func = nn.KLDivLoss(reduction='batchmean')
    n_epochs = config['n_epochs']
    epsilon = 1e-8

    data_filename = config['data_file']
    train_data_loader, test_batch, test_bin, test_edge = load_data(data_filename)

    train_loss_arr = []
    test_loss_arr = []

    directory = config['directory']
    timestamp = datetime.now().strftime("%Y_%m_%d_%H%M%S")    
    run_path = os.path.join(directory, timestamp)
    if not os.path.exists(run_path):
        os.makedirs(run_path)
    log_path = os.path.join(run_path, 'log.txt')

    architecture_path = os.path.join(run_path, "architecture.txt")
    architecture =  '_'.join(str(e) for e in config['model_neurons'])
    with open(architecture_path, 'a') as out_file:
        out_file.write("Neural network architecture\n")
        out_file.write(architecture)

    model_name = "model_" + architecture + "t0_point_1.pt"
    model_path = os.path.join('./files', model_name)

    min_test_loss = 99999999999 # initialize to some large number

    for epoch in range(n_epochs):

        train_loss_cum_sum = 0
        iters_per_epoch = 0

        for tr_batch, tr_bin, _ in train_data_loader:
            tr_batch = tr_batch.squeeze()
            tr_bin = tr_bin.squeeze()
            train_loss = train_step(tr_batch, tr_bin)

            train_loss_cum_sum += train_loss
            iters_per_epoch += 1
        
        avg_train_loss = train_loss_cum_sum/iters_per_epoch
        train_loss_arr.append(avg_train_loss)
        
        test_loss = test_step(test_batch, test_bin)
        test_loss_arr.append(test_loss)

        with open(log_path, 'a') as out_file:
            out_file.write("Epoch: {:d}, train loss: {:f}, test loss: {:f}\n"
                .format(epoch, avg_train_loss, test_loss))

        torch.save(net.state_dict(), model_path)
        scheduler.step(test_loss)

        print(min_test_loss - test_loss)
        if min_test_loss - test_loss > 1e-7:
            epochs_no_improve = 0
            min_test_loss = test_loss
        else:
            epochs_no_improve += 1

        if epoch > 5 and epochs_no_improve == 5:
            print('Early stopping!' )
            break
    
    loss_path = "losses_" + architecture
    loss_path = './files/losses_' + architecture
    np.savez_compressed(loss_path,
                        train_loss = train_loss_arr,
                        test_loss = test_loss_arr
                        )

    fig_path = os.path.join(run_path, 'training_curve.png')
    f = plt.figure()
    plt.plot(train_loss_arr, label='Train Loss')
    plt.plot(test_loss_arr, label='Test Loss')
    plt.xlabel("Epochs")
    plt.ylabel("Loss")
    plt.yscale('log')
    plt.legend()
    f.savefig(fig_path, dpi=300)
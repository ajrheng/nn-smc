# Neural Network Resamplers for Quantum Phase Estimation
This is the repository for the Masters thesis: Neural Network Resamplers for Quantum Phase Estimation at the University of Toronto.

## Setup
Run `pip install -r requirements.txt` to install the prerequisites. It is advised to do inside a virtual environment.

## Configuration Files
All configuration files are found in the `/config` folder. These files contain parameters of the various experiments that are to be carried out.
There are four configuration files:

- `get_data_config.yaml`: Config for generating phase estimation data to train the neural network.
- `train_config.yaml`: Config for training the neural network.
- `run_config_nn.yaml`: Config for running SMC with NN resampler on phase estimation.
- `run_config_bs_yaml`: Config for running SMC with bootstrap (BS) resampler on phase estimation.
- `run_config_lw.yaml`: Config for running SMC with LW resampler on phase estimation.

## Neural Network Resampler

### Generate data
First we need to generate training data. Run
```
python getdata.py
```
A progress bar will be printed on the terminal to display the data generation process. The data file will be compressed and saved as `/files/data.npz`.

### Train network
Next, to train the network, run
```
python train_nn.py
```
Training logs are stored in the `/train_logs` folder. Each time you run the code, a new folder with the timestamp will be created, and in it a
`log.txt` file will be saved which logs the current training epoch and train/test loss. The model will also be saved periodically as `/files/model_[architecture].pt`,
where `[model_filename]` is an argument in the config file.

### Run SMC with NN resampler
Finally, we can run the trained NN resampler with SMC. To run with the NN resampler, 
```
python run.py --resampler nn
```
Similarly, a folder with the timestamp is created in `/run_logs/nn`. Inside, `log.txt` outputs the current run and some basic information. After completion, 
`results.txt` contains the overall MSE and average restarts. Data will be written to './files/[results_filename]', where `[results_filename]` is specified in the config.
Files for errors and cumulative times over all restarts will also be written with `.pkl` extensions.

## Liu-West and Bootstrap Resamplers
For LW and BS, as there is no need for any training, we can proceed to run the resampler directly,
```
python run.py --resampler [lw,bs]
```
Identical to the NN resampler, the same outputs are stored in `/run_logs/lw` or `/run_logs/bs`. Data will be written to `./files/results_[lw/bs].pt`.
# TITLE OF PROJECT
This is the repository for the Masters thesis: _____________.

## Setup
Run `pip install -r requirements.txt` to install the prerequisites. It is advised to do inside a virtual environment.

## Configuration Files
All configuration files are found in the `/config` folder. These files contain parameters of the various experiments that are to be carried out.
There are four configuration files:

- `get_data_config.yaml`: Config for generating phase estimation data to train the neural network.
- `train_config.yaml`: Config for training the neural network.
- `run_config_nn.yaml`: Config for running SMC with NN resampler on phase estimation.
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
`log.txt` file will be saved which logs the current training epoch and train/test loss and the los scurve `training_curve.png`. The model will also be saved periodically as `/files/model.pt`.

### Run SMC with NN resampler
Finally, we can run the trained NN resampler with SMC. To run with the NN resampler, 
```
python run.py --resampler nn
```
Similarly, a folder with the timestamp is created in `/run_logs/nn`. Inside, `log.txt` outputs the current run and some basic information. After completion, 
`results.txt` contains the overall MSE and average restarts, and a plot of the mean and median squared error is shown in `smc_results.png`.

## Liu-West Resampler
For LW, as there is no need for any training, we can proceed to run the resampler directl,
```
python run.py --resampler lw
```
Identical to the NN reasmpler, the same outputs are stored in `/run_logs/lw`.
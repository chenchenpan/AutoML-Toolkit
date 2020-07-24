# Amplify-AutoML-Toolkit
An internship project working at Microsoft

# Demo Data

You are welcome to try out our toolkit with the demo data we provided. The demo data is using the partial Kickstarter dataset collected from [Webrobots.io](https://webrobots.io/projects/).

# Dependencies

- Python 3.0
- TensorFlow>=1.5
- Other required packages are summarized in `requirements.txt`.

# Quick start

## Setup Azure instance

ssh into the instance.

## Download the data and install the dependencies 
```
mkdir ~/projects
cd ~/projects/
git clone https://github.com/chenchenpan/Amplify-AutoML-Toolkit.git

cd ~/projects/Amplify-AutoML-Toolkit/
./vm_setup.sh
```
## Running experiments and monitor with tensorboard

### Define metadata and search space
Use the 'define_metadata_and_search_space.ipynb' to create the metadata file for your tabular data and define the search space to the following experiments. The file we provided contains an example to create metadata and search space for kickstarter demo data. You can modify that and create your own.

### Start Kickstarter experiment
```
screen -S kick
source activate lab
cd ~/projects/Amplify-AutoML-Toolkit/
./run_experiment.sh

```

All the models and results about this experiment will be saved in `~/projects/Amplify-AutoML-Toolkit/outputs`.


### Start tensorboard to monitor WikiTable experiment
```
screen -S tb
source activate lab
cd  ~/projects/Amplify-AutoML-Toolkit/
tensorboard --logdir=outputs
```
To see the tensorboard, in the browser, go to 
[your AzureVM public DNS]:6006


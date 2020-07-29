# Amplify-AutoML-Toolkit
An internship project

# Demo Data

You are welcome to try out our toolkit with the demo data we provided. The demo data is using the partial Kickstarter dataset collected from [Webrobots.io](https://webrobots.io/projects/).

# Dependencies

- Python 3.0
- TensorFlow>=1.11.0
- Other required packages are summarized in `requirements.txt`.

# Quick start

## Setup Azure instance
Start a Data Science Virtual Machine (DSVM). In this repo, we are using Ubuntu 18.04 (Linux VM). For more details about creating the VM, please reference [Creare an Ubuntu Data Science Virtual Machine](https://docs.microsoft.com/en-us/azure/machine-learning/data-science-virtual-machine/dsvm-ubuntu-intro).

Open port (for example, 6000-6010) in the security group for tensorboard. You may find [this blog](https://azadehkhojandi.blogspot.com/2018/11/how-to-run-tensorboard-on-azure-vms.html) helpful.

ssh into the instance.

## Download the data and install the dependencies 
```
mkdir ~/projects
cd ~/projects/
mkdir glove
wget http://nlp.stanford.edu/data/glove.6B.zip
unzip glove.6B.zip -d glove/
rm glove.6B.zip

cd ~/projects/
git clone https://github.com/chenchenpan/Amplify-AutoML-Toolkit.git

cd ~/projects/Amplify-AutoML-Toolkit/
chmod 777 vm_setup.sh
./vm_setup.sh
```
## Running experiments and monitor with tensorboard

### Define metadata and search space
Use the 'define_metadata_and_search_space.ipynb' to create the metadata file for your tabular data and define the search space to the following experiments. The file we provided contains an example to create metadata and search space for kickstarter demo data. You can modify that and create your own.

### Start Kickstarter experiment
```
screen -S demo
source ~/project/envs/lab/bin/activate
cd ~/projects/Amplify-AutoML-Toolkit/

mkdir encoded_data
mkdir outputs

chmod 777 run_encoder.sh
./run_encoder.sh

chmod 777 run_experiment.sh
./run_experiment.sh

```

All the models and results about this experiment will be saved in `~/projects/Amplify-AutoML-Toolkit/outputs`.


### Start tensorboard to monitor experiment
```
screen -S tb
source ~/project/envs/lab/bin/activate
cd  ~/projects/Amplify-AutoML-Toolkit/
tensorboard --logdir=outputs
```
To see the tensorboard, in the browser, go to 
[your AzureVM public DNS]:6006


# AutoML-Toolkit
An internship project in 2020. Updated in 2023.

## Dependencies

- Python 3.0
- TensorFlow>=2.0
- Other required packages are summarized in `requirements.txt`.
- To enable the GPU when running DL model, you may find these articles are helpful:
  - [CUDA Install](https://medium.com/analytics-vidhya/install-cuda-11-2-cudnn-8-1-0-and-python-3-9-on-rtx3090-for-deep-learning-fcf96c95f7a1#id_token=eyJhbGciOiJSUzI1NiIsImtpZCI6ImNhMDA2MjBjNWFhN2JlOGNkMDNhNmYzYzY4NDA2ZTQ1ZTkzYjNjYWIiLCJ0eXAiOiJKV1QifQ.eyJpc3MiOiJodHRwczovL2FjY291bnRzLmdvb2dsZS5jb20iLCJuYmYiOjE2NDI2NjQ2NjcsImF1ZCI6IjIxNjI5NjAzNTgzNC1rMWs2cWUwNjBzMnRwMmEyamFtNGxqZGNtczAwc3R0Zy5hcHBzLmdvb2dsZXVzZXJjb250ZW50LmNvbSIsInN1YiI6IjExMDk2NDYzMjI5MTE3NDE5MDI5OCIsImVtYWlsIjoicGNjLnBrdUBnbWFpbC5jb20iLCJlbWFpbF92ZXJpZmllZCI6dHJ1ZSwiYXpwIjoiMjE2Mjk2MDM1ODM0LWsxazZxZTA2MHMydHAyYTJqYW00bGpkY21zMDBzdHRnLmFwcHMuZ29vZ2xldXNlcmNvbnRlbnQuY29tIiwibmFtZSI6IkNoZW5jaGVuIFBhbiIsInBpY3R1cmUiOiJodHRwczovL2xoMy5nb29nbGV1c2VyY29udGVudC5jb20vYS0vQU9oMTRHaVExSmljZkJVcjI4VXJFOUNXMDM0MUVvZVNJSDREblhTWUNlclRsUT1zOTYtYyIsImdpdmVuX25hbWUiOiJDaGVuY2hlbiIsImZhbWlseV9uYW1lIjoiUGFuIiwiaWF0IjoxNjQyNjY0OTY3LCJleHAiOjE2NDI2Njg1NjcsImp0aSI6ImJmN2M1ZjQ2NmFhZjhjNTVjNDVjNTdkOTJkMDVmMDk1MWYxOTg2MTAifQ.n9bEXVt1sB_SaAsKUxHevx6ITG1y8D9lbXPgPgJFF1YfJ_28xj3kjv77n7Y5MXV5Aw8yHmrWWl0tp6EU90ziGqt4Ep4X-a66M01cMNwcVg2UQcoNC1DPW8VKGHGkINRvHqNNkC67epgJ7oUTMsmn_ObSyID6eX10Lcjs3mBrDpmO3mbt66iT6QDX3qMOyRqIQDBLKYWw6kXzDWGvriBHki2o1gY18R0BA0zOiE8NpQBjcgbS38JF4BcFggjVEGXuqWA4Bvvc9TFebHin0afflQxGYH9zPyPsw1gasgNyCACbmWYJtJDz_haI_xtns6qNwA2eVeBpaiuSdr3jfwTrNA) 

  - [The Ultimate Guide: Ubuntu 18.04 GPU Deep Learning Installation](https://medium.com/@dun.chwong/the-ultimate-guide-ubuntu-18-04-37bae511efb0)

## Quick start

We provide a demo data to help user quick try out this toolkit. The demo data is using a small portion of Kickstarters.com real data which is collected from [Webrobots.io](https://webrobots.io/projects/).

### Step 1: Setup Azure instance
Start a Data Science Virtual Machine (DSVM). In this repo, we are using Ubuntu 18.04 (Linux VM). For more details about creating the VM, please reference [Creare an Ubuntu Data Science Virtual Machine](https://docs.microsoft.com/en-us/azure/machine-learning/data-science-virtual-machine/dsvm-ubuntu-intro).

Open port (for example, 6000-6010) in the security group for tensorboard. You may find [this blog](https://azadehkhojandi.blogspot.com/2018/11/how-to-run-tensorboard-on-azure-vms.html) helpful.

ssh into the instance.

### Step 2: Download the data and install the dependencies 
```
mkdir ~/projects
cd ~/projects/

cd ~/projects/
git clone https://github.com/chenchenpan/AutoML-Toolkit.git

cd ~/projects/AutoML-Toolkit/
chmod 777 vm_setup.sh
./vm_setup.sh
```
### Step 3: Running experiments and monitor with tensorboard

#### Start NN model with GloVe experiment
```
cd ~/projects/AutoML-Toolkit/
mkdir resource
mkdir resource/glove
cd ~/projects/AutoML-Toolkit/resource/glove
wget http://nlp.stanford.edu/data/glove.6B.zip
unzip glove.6B.zip
rm glove.6B.zip

screen -S demo
source ~/project/envs/lab/bin/activate
cd ~/projects/AutoML-Toolkit/demo

chmod 777 run_nn_experiment.sh
./run_nn_experiment.sh

```

#### Start BERT-Tiny experiment
```
screen -S bert
source ~/project/envs/lab/bin/activate
cd ~/projects/AutoML-Toolkit/resource
mkdir bert
wget https://storage.googleapis.com/bert_models/2020_02_20/uncased_L-2_H-128_A-2.zip
unzip uncased_L-2_H-128_A-2.zip
rm uncased_L-2_H-128_A-2.zip

cd ~/projects/AutoML-Toolkit/demo
chmod 777 run_bert_experiment.sh
./run_bert_experiment.sh

```

More BERT models can be found [here](https://github.com/google-research/bert). 
All the models and results about the experiments will be saved in `~/projects/AutoML-Toolkit/demo/outputs`.


#### Start tensorboard to monitor experiment
```
screen -S tb
source ~/project/envs/lab/bin/activate
cd  ~/projects/AutoML-Toolkit/demo
tensorboard --logdir=outputs
```
To see the tensorboard, in the browser, go to 
[your AzureVM public DNS]:6006
(make sure you add the inbound port).
Alternatively, you can monitor your models via local ports. On the remote machine, let's choose port number 8008 and run:
```angular2html
tensorboard --logdir=outputs --port=8008
```
From your local machine, set up ssh port forwarding to one of your unused local ports, for example port 8898:
```angular2html
ssh -NfL localhost:8898:localhost:8008 user@remote
```
Finally, go to `localhost:8898` on your local web browser. The tensorboard interface should pop up.

## Start with your own dataset

Following 3 steps, you can easily use this toolkit to train your own machine learning and deep learning models with any datasets.


### Step 1: Upload prepared datasets
To keep the same organized structure, you can easily copy the `demo` folder and rename it as `my_project`. 

Navigate to `my_project/data/raw_data` directory, replace the `comb_train.tsv`, `comb_dev.tsv` and `comb_test.tsv` under  with your training, validation and test datasets, and delete all the `.json` files (you will create your own metadata file in step 2). 

Optionally, you can delete all the files in `my_project/search_space` directory, since you will define the search space and generate these files in step 2.


### Step 2: Generate metadata and search space files

As an input, you need to provide a `metadata.json` file which describes the datasets structure and data types.

Our toolkit also provides hyperparameter tuning function, so it will need a `search_space.json` file which defines the search space for hyperparameter tuning.

Open `define_metadata_and_search_space.ipynb`, follow it step by step, and you will easily generate these two files. We provide some examples and make sure it bug-free. You can modify it based on your data and models.


### Step 3: Repeat step 3 above in Quick Start

After modifying some arguments (such as `DIR`) in `.sh` files, you can follow the step 3 above in Quick Start to run and monitor the experiments!
cd ~/projects/
pip install virtualenv
mkdir envs && cd envs
virtualenv lab

source lab/bin/activate
cd ~/projects/Amplify-AutoML-Toolkit/
pip install -r requirements.txt 

deactivate


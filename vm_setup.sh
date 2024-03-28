cd ~/Projects/
pip install virtualenv
mkdir envs && cd envs
virtualenv lab

source lab/bin/activate
cd ~/Projects/AutoML-Toolkit/
pip install -r requirements.txt 

deactivate


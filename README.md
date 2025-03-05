# Master Thesis Soruce Code
This repository contains the source code used for the master thesis INSERT NAME written by Erik Mjaanes and Sondre Rogde during the Spring 2025 semester.

### Creating a venv
1. Create a venv 
```sh
python -m venv venv
source venv/bin/activate
```
2. Install packages from requirements.txt
```sh
pip3 install -r requirements.txt
```
3. If you install other packages run this code and push the updated requirements.txt to git
```sh
pip freeze > requirements.txt
```



### Structure
1. To train and tune models, run 
```sh
python src/main_data_generation.py --model-tune model1,model2,model3 --model-generate model1,model2,model3
``` 
from root or with the all-tune or all-train flag set like this to train and tune all available models:
```sh
python src/main_data_generation.py --all-tune=True --all-generate=True
```
2. Hedge models by running
```sh
python something
```
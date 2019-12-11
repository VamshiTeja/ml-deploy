# ml-deploy [![CircleCI](https://circleci.com/gh/VamshiTeja/ml-deploy.svg?style=svg)](https://circleci.com/gh/VamshiTeja/ml-deploy)

Sample Deployment pipeline of Deep Learning Models (Tensorflow). Created a pipeline for deploying machine learning price-prediction service (tensorflow) using flask web framework. 

## Setup 
Requirements: python:3.7, tensorflow:1.13

Clone the repository and install dependencies from requirements.txt
```bash
git clone https://github.com/VamshiTeja/ml-deploy.git
cd ml-deploy
pip install -r requirements.txt
```

## Run 
All the configurations are in a single place: ./config/config.yml. Modify as necessary and run Train/Inference/Flask files.

### Train
```bash
python train.py
```

### Inference
```bash
python inference.py
```

### Flask App
```bash
python app.py
```

# CI and CD
This repo is set up with CircleCI for checking builds,tests (CI) and continuos deployment (CD)

## Deployment
Setup your project in gcloud and create container named "ml-deploy". Then run the following command
```bash
kubectl apply -f kubernetes/
```
Your service is up and check the ip address using the following command
```bash
kubectl get services
```
You can check the logs of pods using 
```bash
kubectl get pods
kubectl logs $POD_NAME
```

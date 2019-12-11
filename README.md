# ml-deploy [![CircleCI](https://circleci.com/gh/VamshiTeja/ml-deploy.svg?style=svg)](https://circleci.com/gh/VamshiTeja/ml-deploy)

Learn Deployment of Machine Learning Models (Tensorflow)

## Setup 
Requirements: python:3.7, tensorflow:1.13
```bash
git clone https://github.com/VamshiTeja/ml-deploy.git
cd ml-deploy
pip install -r requirements.txt
```

## Run 

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
This repo is set up with CircleCI for checking builds,tests and continuos deployment

## Deployment
Setup your project in gcloud and create container. Then create deployment and service
```bash
kubectl apply -f kubernetes/
```


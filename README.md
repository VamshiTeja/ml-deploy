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
All the configurations are in a single place: "./config/config.yml". <br/>
Modify as necessary and run Train/Inference/Flask files.

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

### Create Custom Models

You can easily create custom models with the current setup. Just create new model in "models/$custom_model.py" with format similar to "models/SimpleModel.py". 

Please raise an issue if you face any difficulty!

## Deployment
Setup your gcloud project ($GCLOUD_PROJECT_NAME) and set up GCR for that project. <br/>
First build the docker image and push it to Google Container Registery (GCR).
```bash
docker build -t gcr.io/$GCLOUD_PROJECT_NAME/ml-deploy:v1 .
docker push gcr.io/$GOOGLE_PROJECT_NAME/ml-deploy:v1
```

Create container named "ml-deploy" in google cloud project ($GCLOUD_PROJECT_NAME). Then run the following kubectl commands!
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

## CI and CD
This repo is set up with CircleCI for checking builds,tests (CI) and continuos deployment (CD).

## TODO 
- [ ] Integrate with Tensorflow serving API 
- [ ] Improve init time for Flask Server

.PHONY: dep
dep:
	pip install -r ./requirements.txt

.PHONY: train
train: 
		python train.py

.PHONY: inference
inference: 
		python inference.py

.PHONY: test
test: 
		pytest 
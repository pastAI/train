# train
Repository for pastAI training

The goal was to create an AI that generates a pasta recipe, that only includes the selected ingredients. The idea was to devide the task in three different models:
1. find best fitting ingredients from selection (done)
2. create logical steps for the recipe (to-do)
3. generate recipe text and maybe an image (to-do) 

## 1. Best fitting ingredients model
To create a model it was first needed to create a dataset, with which the model can be trained. For that, a recipe database was used and the combined ingredients were extracted to calculate a score that says how good a combination might fit. 

The model was trained with TensorFlow and optimized with kerastuner.

# Usage with docker
 - Install docker
 - Install docker compose (https://docs.docker.com/compose/install/)
 - Run `docker-compose up -d`
 - Run `docker-compose exec app python simple_train.py` to run a python script
 - Tmp: docker run --gpus all -it --rm -v $PWD:/tmp -w /tmp pastai python ./tuned_train.py

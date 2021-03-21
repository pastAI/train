# pastAI
Repository for pastAI training

The goal was to create an AI that generates a pasta recipe, that only includes the selected ingredients. The idea was to devide the task in three different models:
1. find best fitting ingredients from selection (done)
2. create logical steps for the recipe (to-do)
3. generate recipe text and maybe an image (to-do) 

## 1. Best fitting ingredients model
The dataset used in this model was also created by us. In short, the combined ingredients were extracted out of many recipe datasets to calculate a score that represents how good a combination of ingredients works together. Those scores are saved in a csv, which was the base for the training. 

The model was trained with TensorFlow and optimized with kerastuner as can be seen in /train/tuned_train.py.

Additionally, the whole project should work in a webapp, so everything is hosted and can be found at [pastai.net](pastai.net).

## 2. Logical steps for a recipe (to-do)
The first challenge of creating a recipe is to generate the steps that are needed to prepare a meal. So, for example, when to cut a specific ingredient and what shape would be suitable. We don´t know a dataset that could be used to train such a task, so please let us know if you do.
The steps to continue would be:
- create or find dataset
- develop a proper model structure (input, output, architecture)
- train model
- integrate in webapp

## 3. Generate recipe text (to-do)
After knowing the logical steps, a connected text has to be generated. An idea to solve that would be to use GPT3.

# Usage with docker
 - Install docker
 - Install docker compose (https://docs.docker.com/compose/install/)
 - Run `docker-compose up -d`
 - Run `docker-compose exec app python simple_train.py` to run a python script
 - Tmp: docker run --gpus all -it --rm -v $PWD:/tmp -w /tmp pastai python ./tuned_train.py

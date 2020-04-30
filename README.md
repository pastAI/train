# train
Repository for pastAI training

# Usage with docker
 - Install docker
 - Install docker compose (https://docs.docker.com/compose/install/)
 - Run `docker-compose up -d`
 - Run `docker-compose exec app python simple_train.py` to run a python script
 - Tmp: docker run --gpus all -it --rm -v $PWD:/tmp -w /tmp pastai python ./tuned_train.py
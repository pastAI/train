FROM tensorflow/tensorflow:latest-jupyter
WORKDIR /code
COPY requirements.txt requirements.txt
RUN pip install -r requirements.txt
# our base image
FROM ubuntu:latest

# Install python and pip
# RUN apk add --update py-pip

# Install python modules needed to run the app
# COPY requirements.txt .
# RUN pip install --no-cache-dir -r ./requirements.txt

# https://medium.com/analytics-vidhya/how-to-deploy-machine-learning-models-using-flask-docker-and-google-cloud-platform-gcp-6e7bf1b339d5
# copy code into image and set as working directory
# COPY . /application
# WORKDIR /application
# # install dependencies
# RUN apt-get -y update && \
#     pip install pipenv && \
#     pipenv install --system --deploy

# https://github.com/gw0/docker-keras/blob/master/Dockerfile.py3-tf-cpu

# https://github.com/jeannefukumaru/flask-keras-docker-demo/blob/master/Dockerfile
RUN apt-get update && \
    apt-get -y install sudo
RUN apt-get install -y \
    python3 \
    python3-pip
# RUN apt-get install -y python3-pip python3-dev build-essential
COPY . ./
# COPY . /flask-deployment
# WORKDIR /flask-deployment
RUN sudo -H pip3 install --no-cache-dir -r requirements.txt

# copy files needed to run the app
# COPY emotion_detection_api.py ./
# COPY helpers.py ./
# COPY *.mkl ./
# COPY templates/my-form.html ./templates/

# port number the container should expose for the app
EXPOSE 5000

# run the app
CMD ["python3", "emotion_detection_api.py"]

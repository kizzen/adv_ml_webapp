# Use an official Python runtime as a parent image
FROM python:3.6-slim-stretch
# Set the working directory to /app
WORKDIR /flask
# Copy the current directory contents into the container at /flask
COPY  . /flask
COPY /flask/static/CIFAR/models/ /flask/static/CIFAR/models/
RUN ls -la /flask/static/CIFAR/models/*
COPY /flask/static/MNIST/models /flask/static/MNIST/models/
COPY /flask/static/CIFAR/DeepFool /flask/static/CIFAR/DeepFool
COPY /flask/static/MNIST/DeepFool /flask/static/MNIST/DeepFool
COPY /flask/static/CIFAR/FGSM /flask/static/CIFAR/FGSM
COPY /flask/static/MNIST/FGSM /flask/static/MNIST/FGSM
COPY /flask/static/CIFAR/JSMA/ /flask/static/CIFAR/JSMA
COPY /flask/static/MNIST/JSMA /flask/static/MNIST/JSMA 
# Install any needed packages specified in requirements.txt
ADD requirements.txt /
RUN pip  --no-cache-dir install -r /requirements.txt
# Make port 80 available to the world outside this container
EXPOSE 80
# Run advml_app.py when the container launches
CMD ["python", "flask/advml_app.py"]

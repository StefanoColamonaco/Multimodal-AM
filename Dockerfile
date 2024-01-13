# Use a base image of py 3.6
FROM python:3.6

# Set the working directory
WORKDIR /src

# Copy the dataset files to the container
COPY . /src

# Install any necessary dependencies
RUN apt-get update

RUN apt-get install -y python3-pip

# Upgrade pip
RUN pip3 install --upgrade pip

RUN pip3 install numpy==1.19.5

# Install python dependencies
RUN pip3 install -r requirements.txt
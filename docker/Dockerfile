# Use the official Python image as the base image
FROM python:3.10.9 

# Mount the workspace (top directory containing the setup.py) inside the container
WORKDIR /workspace

# Copy the contents of your package to the container
COPY . /workspace

RUN pip install -e .

RUN pip install --no-cache-dir    \
    hydra-core==1.3.1             \
    pytorch-lightning==1.9.4      \
    pytorch==1.13.1               \
    matplotlib                    \
    tensorboard                   \
    tensorflow
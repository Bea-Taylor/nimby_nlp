# Use a base image with miniforge (includes mamba)
FROM condaforge/miniforge3:24.3.0-0

# Set the working directory inside the container
WORKDIR /app

# Copy the environment.yml file first to leverage Docker cache
# If environment.yml changes, this layer (and subsequent ones) will rebuild.
COPY environment.yml .

# Create the conda environment using mamba (generally faster)
# The `conda init` is usually handled by the base image, allowing `conda activate`.
# `source activate` is for older conda versions or specific shell setups.
RUN mamba env create -f environment.yml && \
    mamba clean --all -y

# Copy the rest of your application code
# This layer will only rebuild if your application code changes, not the environment.
COPY . .

EXPOSE 8888
# Set the default command to run when the container starts.
# We explicitly activate the 'nlp_env_analysis' environment
# before running the jupyter notebook command.
CMD ["mamba", "run", "-n", "nlp_env_analysis", "jupyter", "notebook", "--ip=0.0.0.0", "--port=8888", "--allow-root","--ServerApp.token=''"]

# Expose the Jupyter port
EXPOSE 8888
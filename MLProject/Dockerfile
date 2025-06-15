FROM continuumio/miniconda3

WORKDIR /app

# Copy only conda.yaml first for better layer caching
COPY conda.yaml .

# Update conda and create environment
RUN conda update -n base -c defaults conda && \
    conda env create -f conda.yaml && \
    conda clean -afy

# Copy the rest of the application
COPY . .

# Set PATH to use the conda environment by default
ENV PATH /opt/conda/envs/mlflow-env/bin:$PATH

# Expose the port
EXPOSE 5005

# Run the script
ENTRYPOINT ["python", "modeling.py"]
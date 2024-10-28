# Use the official Julia 1.10 image as a base
FROM julia:1.10

# Install any additional packages you need
RUN apt-get update && apt-get install -y \
    curl \
    wget \
    git \
    && rm -rf /var/lib/apt/lists/*

# Set up the Julia environment
RUN julia -e 'using Pkg; Pkg.instantiate(); Pkg.precompile()'

# Expose the necessary ports
EXPOSE 8888

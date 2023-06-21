#!/bin/sh
pip install -r requirements.txt && \
conda install -c conda-forge cudatoolkit=11.2 cudnn=8.1.0 -y && \
cd && echo "export LD_LIBRARY_PATH=$LD_LIBRARY_PATH:$CONDA_PREFIX/lib/" >> ~/.bashrc && \
mkdir -p $CONDA_PREFIX/etc/conda/activate.d && \
echo 'export LD_LIBRARY_PATH=$LD_LIBRARY_PATH:$CONDA_PREFIX/lib/' > $CONDA_PREFIX/etc/conda/activate.d/env_vars.sh 

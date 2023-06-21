Usage:
    1. Create conda environmen
        conda create -n deepfl_eval
    2. Activate conda environmen
        conda activate deepfl_eval
    3. Cd to requirements
        cd requirements
    4. Install all required packages
        bash requirements.sh
    5. Download weights and put to corresponding folders
        Links of weights are in models/link.txt
    6. Download PCA models and put to corresponding folders
        Links of weights are in models/link.txt
    
    To Use Evaluater for a single image run DeepFL.py 
        Giving the path of image
            -d (--data_path) path_of_image
        If you want to visualize it
            -v (--visualize) true
    To Use Evaluater for all benchmarks
        Open jupyter notebook and run Evaluator.ipynb
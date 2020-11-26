# Amazon Review Generator
Trains an encoder-decoder model from pretrained Bert and GPT2 checkpoints.

## Model training
Run the `training.ipynb` Jupyter notebook file.

### Dependencies
Create Conda environment from yaml file.

`conda env create --file environment.yml`

Enable CUDA
```
module load cuda10.2/toolkit/10.2.89
module load CUDA/10.2.89-GCC-6.4.0-2.28
```

### Configuration
Constants 

`MODEL_SAVE_PATH`: model save directory path

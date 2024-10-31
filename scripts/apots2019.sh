#!/bin/bash

# Download the dataset using kagglehub
python -c "
import kagglehub
path = kagglehub.dataset_download('mariaherrerot/aptos2019')
print('Path to dataset files:', path)
"
import kagglehub
import os
# Download latest version
os.environ['KAGGLEHUB_CACHE'] = os.getcwd()
path = kagglehub.dataset_download("paultimothymooney/chest-xray-pneumonia")
print("Path to dataset files:", path)
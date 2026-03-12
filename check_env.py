import sys
import tensorflow as tf
import pandas as pd
import numpy as np
import sklearn

print("Python:", sys.version)
print("TensorFlow:", tf.__version__)
print("Pandas:", pd.__version__)
print("NumPy:", np.__version__)
print("Scikit-learn:", sklearn.__version__)
print("GPUs:", tf.config.list_physical_devices("GPU"))
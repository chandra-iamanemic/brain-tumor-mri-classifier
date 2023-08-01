#%%
import tensorflow as tf


# %%

# Avoid OOM errors by setting GPU Memory Consumption Growth as True
gpus = tf.config.experimental.list_physical_devices('GPU')

for gpu in gpus:
    tf.config.experimental.set_memory_growth(gpu, True)
    
tf.config.list_physical_devices('GPU')

#%%
import cv2
import imghdr


#%% [markdown]
# Import necessary libraries

#%%
import tensorflow as tf
from tensorflow.keras.layers import  Dense, Flatten, MaxPooling2D 
from tensorflow.keras.models import Model
from tensorflow.keras.applications.vgg16 import VGG16
from tensorflow.keras.applications.vgg16 import preprocess_input
from tensorflow.keras.preprocessing.image import ImageDataGenerator
from tensorflow.keras import optimizers
from tensorflow.keras.callbacks import ModelCheckpoint
import cv2

from datetime import datetime
import matplotlib.pyplot as plt
import matplotlib.image as img
import os
import numpy as np
import shutil
import warnings
warnings.filterwarnings("ignore")

#%%


# Avoid OOM errors by setting GPU Memory Consumption Growth as True
gpus = tf.config.experimental.list_physical_devices('GPU')
print("Num GPUs Available: ", len(gpus))

for gpu in gpus:
    tf.config.experimental.set_memory_growth(gpu, True)


tf.test.is_gpu_available()

#%% [markdown]

# Take the raw data folder and loop through the yes and no folders one by one
# Read all the file names in the yes and no folders and split them into test and train sets
# Copy the file names into train and test folders for yes and no respectively


#%%
raw_data_path = 'raw dataset'
categories_dir = ['yes', 'no']
root_dir = 'images'

raw_data_path
# for i in categories_dir:
#     if not os.path.exists(f'{root_dir}/train/{i}'):
#         os.makedirs(f'{root_dir}/train/{i}')
#     if not os.path.exists(f'{root_dir}/test/{i}'):
#         os.makedirs(f'{root_dir}/test/{i}')

#     source = raw_data_path + '/' + i

#     allFileNames = os.listdir(source)

#     np.random.shuffle(allFileNames)

#     test_split_ratio = 0.3
#     test_len = int(len(allFileNames)*test_split_ratio)

#     train_FileNames = allFileNames[:-test_len]
#     test_FileNames = allFileNames[-test_len:]

#     for name in train_FileNames:
#         pass
#         shutil.copy(f'{source}/{name}', f'{root_dir}/train/{i}')

#     for name in test_FileNames:
#         shutil.copy(f'{source}/{name}', f'{root_dir}/test/{i}')


# %%
#Setting dataset path for train and test sets 
train_path = f"{root_dir}/train"
test_path = f"{root_dir}/test"


# %% [markdown]
# Create a subplot of 2x4 images
# plot images from train set and get an overview of the two categories
# plot row 1 with 4 images from category 1
# plot row 2 with 4 images from category 2

#%%
fig, ax = plt.subplots(2, 4, figsize=(30,15))
fig.suptitle("Brain MRI Images (Tumor yes or no)", fontsize=40)

for i in range(2):
    for j in range(4):
        img_source = f"{root_dir}/train/{categories_dir[i]}"
        img_files = os.listdir(img_source)
        np.random.shuffle(img_files)
        img_current = f"{img_source}/{img_files[j]}"

        image = img.imread(img_current)
        ax[i,j].imshow(image)
        ax[i,j].set_title(categories_dir[i], fontsize=20)
        
# %% [markdwon]
# Image Data Generator is used to augment existing images to create variations of the image
# By using this method we can generate a bigger dataset than the one we have by augmenting and creating new images from existing ones


train_datagen = ImageDataGenerator(
    preprocessing_function=preprocess_input,
    rotation_range=90,
    width_shift_range=0.2,
    height_shift_range=0.2,
    shear_range=0.2,
    zoom_range=0.2,
    horizontal_flip=True,
    vertical_flip= True,

    fill_mode='nearest')

# test Data Augmentation 
test_datagen = ImageDataGenerator(
    preprocessing_function=preprocess_input,
    rotation_range=90,
    width_shift_range=0.2,
    height_shift_range=0.2,
    shear_range=0.2,
    zoom_range=0.2,
    horizontal_flip=True,
    vertical_flip= True,
    fill_mode='nearest')

# %%
train_set = train_datagen.flow_from_directory(train_path,
                                                 target_size = (224, 224),
                                                 batch_size = 32,
                                                 class_mode = 'binary')
test_set = test_datagen.flow_from_directory(test_path,
                                            target_size = (224, 224),
                                            batch_size = 4,
                                            class_mode = 'binary')
no_of_validation_batches = len(test_set)
validation_steps = np.ceil(no_of_validation_batches)

#%%
sample_batch = test_set.next()

sample_images = sample_batch[0]
sample_labels = sample_batch[1]
for i,j in zip(sample_images, sample_labels):
    i = cv2.cvtColor(i, cv2.COLOR_BGR2RGB)
    cv2.imshow(f"{j}",i)
    cv2.waitKey(0)
cv2.destroyAllWindows()
    


# %%
IMAGE_SIZE = [224, 224, 3]
base_model = VGG16(input_shape=IMAGE_SIZE, weights='imagenet', include_top=False)


#%% [markdown]
# input one batch of our images into the base model and see how it is converted after passing through
# We can check the shape of the output from the last layer
# we will get extracted features once we pass an image through the base model

sample_images.shape

features_from_base_model = base_model(sample_images)
print(features_from_base_model.shape)

# %% [markdown]
# parameters of the VGG16 model are frozen to prevent them from being updated during training. 
# By freezing the model we make sure that it doesn't retrain the layers
# We want to retain whatever it has already learnt 

base_model.trainable = False
base_model.summary()

#%% [markdown]
# We are adding our own trainable layer on top of the base model 
# We can input the feature batch we retrieved from out previous layer to this and check the output shape

# %%
from keras import Sequential
from keras.layers import Conv2D, MaxPooling2D, Flatten, Dense, GlobalAveragePooling2D

global_average_layer = GlobalAveragePooling2D()
features_from_average_layer = global_average_layer(features_from_base_model)

print(features_from_average_layer.shape)

#%% [markdown]
# We are adding the final softmax layer that will give us the prediction

#%%

prediction_layer = Dense(1, activation='sigmoid')
features_from_prediction_layer = prediction_layer(features_from_average_layer)
print(features_from_prediction_layer.shape)

#%% [markdown]
# We form our final model using 
# Base Model
# the flattened  global average layer
# the final prediction dense layer

#%%


model = Sequential([
   base_model,
   global_average_layer,
   prediction_layer
])

#%% [markdown]
# We initialize our parameters for training such as the optimizer, loss and learning rates
adam = optimizers.Adam()
model.compile(loss='binary_crossentropy',
              optimizer=adam,
              metrics=['accuracy'])

model.summary()

# %%
checkpoint = ModelCheckpoint(filepath='mymodel.h5', 
                               verbose=2, 
                               save_best_only=True)
callbacks = [checkpoint]
start = datetime.now()

history = model.fit(train_set,
                      validation_data=test_set,
                      epochs=10,
                      validation_steps=validation_steps,
                      callbacks=callbacks , 
                      verbose=2)


duration = datetime.now() - start
print("Training completed in time: ", duration)

#%%


def plot_accuracy_loss(history):
   
   fig = plt.figure(figsize= (10,5))

   #Accuracy Plot
   plt.subplot(221)
   plt.plot(history.history['accuracy'], 'bo--', label= "acc")
   plt.plot(history.history['val_accuracy'], 'ro--', label= "val_acc")
   plt.title("Training Accuracy vs Validation Accuracy")
   plt.ylabel("accuracy")
   plt.xlabel("epochs")
   plt.legend()


   #Loss Function Plot
   plt.subplot(222)
   plt.plot(history.history['loss'], 'bo--', label= "loss")
   plt.plot(history.history['val_loss'], 'ro--', label= "val_loss")
   plt.title("Training Loss vs Validation Loss")
   plt.ylabel("loss")
   plt.xlabel("epochs")
   plt.legend()

   plt.show()


plot_accuracy_loss(history)

# %%
# The base_model parameters are unfrozen to allow fine-tuning of the entire model
base_model.trainable = True
model.summary()

# %%
# Adam optimizer is created with a lower learning rate (1e-5) for fine-tuning.
adam = optimizers.Adam(1e-5)
model.compile(loss='binary_crossentropy',
              optimizer=adam,
              metrics=['accuracy'])

#%%

checkpoint = ModelCheckpoint(filepath='mymodel.h5', 
                               verbose=2, save_best_only=True)
callbacks = [checkpoint]
start = datetime.now()

history_finetuned = model.fit(
                      train_set,
                      validation_data=test_set,
                      epochs=50,
                      validation_steps=validation_steps,
                      callbacks=callbacks ,verbose=2)


duration = datetime.now() - start
print("Training completed in time: ", duration)

# %%
plot_accuracy_loss(history_finetuned)

# %%
loss, acc = model.evaluate(test_set)

print(f"The final Accuracy of our model on our test set is : {acc}%")

# %%

from keras.applications.resnet50 import ResNet50
from keras.preprocessing import image
from keras.applications.resnet50 import preprocess_input
from keras.models import Model
from keras.layers import Dense
from keras.optimizers import Adam, SGD
from keras.callbacks import EarlyStopping, ModelCheckpoint
from sklearn.preprocessing import LabelEncoder
from sklearn.model_selection import train_test_split
from sklearn.utils import class_weight

import numpy as np
import tensorflow as tf
from utils import load_imgs
import os
os.environ['CUDA_VISIBLE_DEVICES'] = '1'

# vram limit for eficiency
config = tf.ConfigProto()
config.gpu_options.allow_growth = True
sess = tf.Session(config=config)

# define class names
classes = ['lecoq', 'crocs', 'vans', 'nike', 'adidas', 'reebok', 'sbenu',
           'puma', 'drmartens', 'zeepseen', 'descente', 'converse', 'newbalance', 'barefoot']

# define our image size (width, height, channels)
target_size = (224, 224)

# load unprocessed images
data_x, data_y = load_imgs(datapath='/home/tkdrlf9202/Datasets/shoes_classification',
                           classes=classes, target_size=target_size)
print(data_x.shape)
# preprocess images for the model
data_x = preprocess_input(data_x)


# preprocess labels with scikit-learn LabelEncoder
label_encoder = LabelEncoder()
label_encoder.fit(data_y)
data_y_int = label_encoder.transform(data_y)

# calculate class weight
cls_weight = class_weight.compute_class_weight('balanced', np.unique(data_y_int), data_y_int)

# split the data to tran & valid data
data_x_train, data_x_valid, data_y_train, data_y_valid = train_test_split(data_x, data_y_int,
                                                                          test_size=0.2)

# define image data generator for on-the-fly augmentation
generator = image.ImageDataGenerator(zca_whitening=False, rotation_range=10,
                                     width_shift_range=0.1, height_shift_range=0.1,
                                     shear_range=0.02, zoom_range=0.1,
                                     channel_shift_range=0.05, horizontal_flip=True)

# fit the generator (required if zca_whitening=True)
generator.fit(data_x)

# load the model without output layer for fine-tuning
model_baseline = ResNet50(weights='imagenet', include_top=False, pooling='avg')
features = model_baseline.output
# add output layer
predictions = Dense(len(classes), activation='softmax')(features)

# define the model
model_shoes = Model(inputs=model_baseline.input, outputs=predictions)

# freeze the resnet layers except the last 2 blocks
for layer in model_baseline.layers[:154]:
    layer.trainable = False

# compile
opt = Adam(lr=0.0001)
opt_sgd = SGD(lr=5*1e-4, momentum=0.9)
model_shoes.compile(optimizer=opt, loss='sparse_categorical_crossentropy',
                    metrics=['accuracy'])
model_shoes.summary()

# train
batch_size = 32
early_stop = EarlyStopping(monitor='val_loss', patience=10)
ckpt = ModelCheckpoint(filepath='ckpt_adam_nocrop_norm-{epoch:02d}-{val_loss:.2f}.h5', verbose=1, save_best_only=True, mode='auto')
model_shoes.fit_generator(generator.flow(data_x_train, data_y_train, batch_size=batch_size, shuffle=True),
                          epochs=1000,
                          steps_per_epoch=int(data_x.shape[0])/batch_size,
                          class_weight=cls_weight,
                          validation_data=(data_x_valid, data_y_valid),
                          callbacks=[early_stop, ckpt])

# save the model
model_shoes.save('model_shoes_freeze_2_adam_clsweight_nocrop_norm.h5')
print('model saved')
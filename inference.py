from keras.models import load_model
from utils import load_imgs
from sklearn.preprocessing import LabelEncoder
from sklearn.metrics import accuracy_score
from keras.applications.resnet50 import preprocess_input
import numpy as np
import tensorflow as tf

# vram limit for eficiency
config = tf.ConfigProto()
config.gpu_options.allow_growth = True
sess = tf.Session(config=config)

############# same as training phase ##############
# define class names
classes = ['lecoq', 'crocs', 'vans', 'nike', 'adidas', 'reebok', 'sbenu',
           'puma', 'drmartens', 'zeepseen', 'descente', 'converse', 'newbalance', 'barefoot']
# define our image size
target_size = (224, 224)
###################################################

# load the test data
# currently using debug data(SAME as training data, so it's cheating)
data_x, data_y = load_imgs(datapath='/home/tkdrlf9202/Datasets/shoes_test', classes=classes, target_size=target_size)

############# same as training phase ##############
# preprocess images for the model
data_x = preprocess_input(data_x)
# preprocess labels with scikit-learn LabelEncoder
label_encoder = LabelEncoder()
label_encoder.fit(data_y)
data_y_int = label_encoder.transform(data_y)
###################################################

# load the model
model_shoes = load_model('ckpt_adam_nocrop_slow-05-0.80.h5')

# predict with the model
preds = model_shoes.predict(data_x, batch_size=1, verbose=1)
preds = np.argmax(preds, axis=1)
print('')
print(accuracy_score(y_true=data_y_int, y_pred=preds))

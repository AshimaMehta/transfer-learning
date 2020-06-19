Using MobileNet for Monkey Classifer
Loading the MobileNet Model
#Freeze all layers except the top 4, as we'll only be training the top 4

In [1]:
from keras.applications import MobileNet

img_rows, img_cols = 224, 224 

# Re-loads the MobileNet model without the top or FC layers
MobileNet = MobileNet(weights = 'imagenet', 
                 include_top = False, 
                 input_shape = (img_rows, img_cols, 3))

for layer in MobileNet.layers:
    layer.trainable = False
    
# printing the layers 
f
    print(str(i) + " "+ layer.__class__.__name__, layer.trainable)
Using TensorFlow backend.
0 InputLayer False
1 ZeroPadding2D False
2 Conv2D False
3 BatchNormalization False
4 ReLU False
5 DepthwiseConv2D False
6 BatchNormalization False
7 ReLU False
8 Conv2D False
9 BatchNormalization False
10 ZeroPadding2D False
11 DepthwiseConv2D False
12 BatchNormalization False
13 ReLU False
14 Conv2D False
15 ReLU False
16  DepthwiseConv2D False
17 BatchNormalization False
18  ReLU False
19  Conv2D False
20 BatchNormalization False
21 ReLU False
22 ZeroPadding2D False
23 DepthwiseConv2D False
24  BatchNormalization False
25 ReLU False
26 Conv2D False
27 BatchNormalization False 
28 ReLU False
29 DepthwiseConv2D False
30 BatchNormalization False
31 ReLU False
32 Conv2D False
33 BatchNormalization False
34 ReLU False
35 ZeroPadding2D False
36 DepthwiseConv2D False
37 BatchNormalization False
38 ReLU False
39 Conv2D False
40 BatchNormalization False
41 ReLU False
42 DepthwiseConv2D False
43 BatchNormalization False
44 ReLU False
45 Conv2D False
46 BatchNormalization False
47 ReLU False

function to return  FC Head
In [2]:
def lw(bottom_model, num_classes):
    """creates the top or head of the model that will be 
    placed ontop of the bottom layers"""

    top_model = bottom_model.output
    top_model = GlobalAveragePooling2D()(top_model)
    top_model = Dense(1024,activation='relu')(top_model)
    top_model = Dense(1024,activation='relu')(top_model)
    top_model = Dense(512,activation='relu')(top_model)
    
 
In [3]:
from keras.models import Sequential
from keras.layers import Dense, Dropout, Activation, Flatten, GlobalAveragePooling2D
from keras.layers import Conv2D, MaxPooling2D, ZeroPadding2D
from keras.layers.normalization import BatchNormalization
from keras.models import Model

num_classes = 5

FC_Head = lw(MobileNet, num_classes)

model = Model(inputs = MobileNet.input, outputs = FC_Head)

Monkey Breed Dataset
In [4]:
from keras.preprocessing.image import ImageDataGenerator

train_data_dir = 'c_mobilenet/c_mobilenet/train/'
validation_data_dir = 'c_mobilenet/c_mobilenet/val/'

# Let's use some data augmentaiton 
train_datagen = ImageDataGenerator(
      rescale=1./255,
      rotation_range=45,
      width_shift_range=0.3,
      height_shift_range=0.3,
      horizontal_flip=True,
      fill_mode='nearest')
 
validation_datagen = ImageDataGenerator(rescale=1./255)
 
batch_size = 32
 
train_generator = train_datagen.flow_from_directory(
        train_data_dir,
        target_size=(img_rows, img_cols),
        batch_size=batch_size,
        class_mode='categorical')
 
validation_generator = validation_datagen.flow_from_directory(
        validation_data_dir,
        target_size=(img_rows, img_cols),
        batch_size=batch_size,
        class_mode='categorical')
Found 389 images belonging to 5 classes.
Found 25 images belonging to 5 classes.
Training out Model
Note we're using checkpointing and early stopping
In [5]:
from keras.optimizers import RMSprop
from keras.callbacks import ModelCheckpoint, EarlyStopping

                     
checkpoint = ModelCheckpoint("c_mobileNet.h5",
                             monitor="val_loss",
                             mode="min",
                             save_best_only = True,
                             verbose=1)

earlystop = EarlyStopping(monitor = 'val_loss', 
                          min_delta = 0, 
                          patience = 3,
                          verbose = 1,
                          restore_best_weights = True)

# we put our call backs into a callback list
callbacks = [earlystop, checkpoint]

# We use a very small learning rate 
model.compile(loss = 'categorical_crossentropy',
              optimizer = RMSprop(lr = 0.001),
              metrics = ['accuracy'])

# Enter the number of training and validation samples here
nb_train_samples = 1099
nb_validation_samples = 346

# We only train 5 EPOCHS 
epochs = 7
batch_size = 16

history = model.fit_generator(
    train_generator,
    steps_per_epoch = nb_train_samples // batch_size,
    epochs = epochs,
    callbacks = callbacks,
    validation_data = validation_generator,
    validation_steps = nb_validation_samples // batch_size)


Restoring model weights from the end of the best epoch

Epoch 00005: val_loss did not improve from 0.8613
Epoch 00005: early stopping
Loading our classifer
In [6]:
from keras.models import load_model

classifier = load_model('c_mobileNet.h5')
Testing our classifer on some test images
In [7]:
import os
import cv2
import numpy as np
from os import listdir
from os.path import isfile, join

monkey_breeds_dict = {"[0]": "joe", 
                      "[1]": "john-el",
                      "[2]": "maddy",
                      "[3]": "mada",
                      "[4]": "marty"}

monkey_breeds_dict_n = {"n0": "joe", 
                        "n1": "john-el",
                        "n2": "maddy",
                        "n3": "mada",
                        "n4": "marty"}

def draw_test(name, pred, im):
    monkey = monkey_breeds_dict[str(pred)]
    BLACK = [0,0,0]
    expanded_image = cv2.copyMakeBorder(im, 70, 0, 0, 100 ,cv2.BORDER_CONSTANT,value=BLACK)
    cv2.putText(expanded_image, monkey, (20, 60) , cv2.FONT_HERSHEY_SIMPLEX,1, (0,0,233), 2)
    cv2.imshow(name, expanded_image)

def getRandomImage(path):
    """function loads a random images from a random folder in our test path """
    folders = list(filter(lambda x: os.path.isdir(os.path.join(path, x)), os.listdir(path)))
    random_directory = np.random.randint(0,len(folders))
    path_class = folders[random_directory]
    print("Class - " + monkey_breeds_dict_n[str(path_class)])
    file_path = path + path_class
    file_names = [f for f in listdir(file_path) if isfile(join(file_path, f))]
    random_file_index = np.random.randint(0,len(file_names))
    image_name = file_names[random_file_index]
    return cv2.imread(file_path+"/"+image_name)    

for i in range(0,10):
    input_im = getRandomImage("c/c/val/")
    input_original = input_im.copy()
    input_original = cv2.resize(input_original, None, fx=0.5, fy=0.5, interpolation = cv2.INTER_LINEAR)
    
    input_im = cv2.resize(input_im, (224, 224), interpolation = cv2.INTER_LINEAR)
    input_im = input_im / 255.
    input_im = input_im.reshape(1,224,224,3) 
    
    # Get Prediction
    res = np.argmax(classifier.predict(input_im, 1, verbose = 0), axis=1)
    
    # Show image with predicted class
    draw_test("Prediction", res, input_original) 
    cv2.waitKey(0)

cv2.destroyAllWindows()
Class - joe
Class - john-el
Class - maddy 
Class - mada
Class - Marty 

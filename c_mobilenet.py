from keras.applications import MobileNet
img_rows, img_cols = 224, 224
MobileNet = MobileNet(weights = 'imagenet', include_top = False, input_shape = (img_rows, img_cols, 3))
for layer in MobileNet.layers:
    layer.trainable = False
for (i,layer) in enumerate(MobileNet.layers):
    print(str(i) + " "+ layer.__class__.__name__, layer.trainable)
def lw(bottom_model, num_classes):
    top_model = bottom_model.output
    top_model = GlobalAveragePooling2D()(top_model)
    top_model = Dense(1024,activation='relu')(top_model)
    top_model = Dense(1024,activation='relu')(top_model)
    top_model = Dense(512,activation='relu')(top_model)
    top_model = Dense(num_classes,activation='softmax')(top_model)
    return top_model
from keras.models import Sequential
from keras.layers import Dense, Dropout, Activation, Flatten, GlobalAveragePooling2D
from keras.layers import Conv2D, MaxPooling2D, ZeroPadding2D
from keras.layers.normalization import BatchNormalization
from keras.models import Model
num_classes = 5
FC_Head = lw(MobileNet, num_classes)
model = Model(inputs = MobileNet.input, outputs = FC_Head)
print(model.summary())
from keras.preprocessing.image import ImageDataGenerator
train_data_dir = 'c_mobilenet/c_mobilenet/train/'
validation_data_dir = 'c_mobilenet/c_mobilenet/val/'
train_datagen = ImageDataGenerator(rescale=1./255,rotation_range=45,width_shift_range=0.3,height_shift_range=0.3,horizontal_flip=True,fill_mode='nearest')
validation_datagen = ImageDataGenerator(rescale=1./255)
batch_size = 32
train_generator = train_datagen.flow_from_directory(train_data_dir,target_size=(img_rows, img_cols),batch_size=batch_size,class_mode='categorical')
validation_generator = validation_datagen.flow_from_directory(validation_data_dir,target_size=(img_rows, img_cols), batch_size=batch_size,class_mode='categorical')
from keras.optimizers import RMSprop
from keras.callbacks import ModelCheckpoint, EarlyStopping
checkpoint = ModelCheckpoint("c_mobileNet.h5",monitor="val_loss",mode="min",save_best_only = True,verbose=1)
earlystop = EarlyStopping(monitor = 'val_loss', min_delta = 0, patience = 3,verbose = 1,restore_best_weights = True)
callbacks = [earlystop, checkpoint]
model.compile(loss = 'categorical_crossentropy',optimizer = RMSprop(lr = 0.001),metrics = ['accuracy'])
nb_train_samples = 1019
nb_validation_samples = 372
epochs = 7
batch_size = 16
history = model.fit_generator(train_generator,steps_per_epoch = nb_train_samples // batch_size,epochs = epochs,callbacks = callbacks,validation_data = validation_generator,validation_steps = nb_validation_samples // batch_size)
from keras.models import load_model
classifier = load_model('c_mobileNet.h5')
import os
import cv2
import numpy as np
from os import listdir
from os.path import isfile, join
monkey_breeds_dict = {"[0]": "marty", "[1]": "john-el","[2]": "joe","[3]": "mada","[4]": "maddy"}
monkey_breeds_dict_n = {"n0": "marty", "n1": "john-el","n2": "joe","n3": "mada","n4": "maddy"}
def draw_test(name, pred, im):
    monkey = monkey_breeds_dict[str(pred)]
    BLACK = [0,0,0]
    expanded_image = cv2.copyMakeBorder(im, 80, 0, 0, 100 ,cv2.BORDER_CONSTANT,value=BLACK)
    cv2.putText(expanded_image, monkey, (20, 60) , cv2.FONT_HERSHEY_SIMPLEX,1, (0,0,255), 2)
    cv2.imshow(name, expanded_image)
def getRandomImage(path):
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
    input_im = getRandomImage("c_mobilenet/c_mobilenet/val/")
    input_original = input_im.copy()
    input_original = cv2.resize(input_original, None, fx=0.5, fy=0.5, interpolation = cv2.INTER_LINEAR)
    input_im = cv2.resize(input_im, (224, 224), interpolation = cv2.INTER_LINEAR)
    input_im = input_im / 255.
    input_im = input_im.reshape(1,224,224,3)
    res = np.argmax(classifier.predict(input_im, 1, verbose = 0), axis=1)
    draw_test("Prediction", res, input_original) 
    cv2.waitKey(0)
cv2.destroyAllWindows()

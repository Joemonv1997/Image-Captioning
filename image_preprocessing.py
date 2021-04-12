import tensorflow as tf
import pickle
from tensorflow.keras import *
import os
import pathlib
folder="D:/Image_Captioning/Flicker8k_Dataset"
vgg=applications.vgg16.VGG16()
model=Model(vgg.input,vgg.layers[-2].output)
print(model.summary())
print("/n")
print(vgg)
features=dict()
for i in os.listdir(folder):
    img=preprocessing.image.load_img(os.path.join(folder,i),target_size=(224,224))
    img=preprocessing.image.img_to_array(img)
    img=img.reshape((1,img.shape[0],img.shape[1],img.shape[2]))
    img_pre=applications.vgg16.preprocess_input(img)
    predict=model.predict(img_pre,verbose=0)
    print(i,predict)
    ud=i.split(".")[0]
    features[ud]=predict
print(features)
with open("features.pkl","wb") as fs:
    pickle.dump(features,fs)


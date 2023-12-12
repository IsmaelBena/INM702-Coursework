import numpy as np
from numpy.random import default_rng
import tensorflow as tf
from tensorflow import keras
import os
import PIL
import pandas as pd

filepath='.'
filedir=os.path.join(os.getcwd(),filepath,'Vegetable Images')
columns=['img_name','label']
fruitarray=[]
for i in os.listdir(filedir): #train, test valid
    filenames=os.path.join(filedir,i)
    # print(i)
    for j in (os.listdir(filenames)):
        fruitnames=os.path.join(filenames,j)
        for iter,k in enumerate(os.listdir(fruitnames)):
            imgnames=os.path.join(fruitnames,k)
            if(i=='train'):
                train_df=pd.DataFrame(columns=columns)
                train_df['img_name'][iter]=k
                train_df['label'][iter]=j
            elif(i=='test'):
                test_df=pd.DataFrame(columns=columns)
                test_df['img_name'][iter]=k
                test_df['label'][iter]=j
            elif(i=='validation'):
                valid_df=pd.DataFrame(columns=columns)
                valid_df['img_name'][iter]=k
                valid_df['label'][iter]=j
            # print(imgnames)
# print(filenames_train)

print(valid_df.head(25))
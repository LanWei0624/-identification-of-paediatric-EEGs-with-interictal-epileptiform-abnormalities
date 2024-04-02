import os
from tensorflow.keras.preprocessing import image_dataset_from_directory
from tensorflow.keras.layers import Dense, Flatten
from tensorflow.keras.models import Sequential, Model
from tensorflow.keras.callbacks import ModelCheckpoint
from tensorflow.keras import regularizers
#from livelossplot import PlotLossesKeras
from tensorflow.keras.applications.vgg19 import VGG19




RANDOM_SEED = 42

path_ = 'scratch/CHI_figure/Train/'

#Preparing images for model training --- image size = 256 x 256
train_batch = 64
test_batch = 64
train_set= image_dataset_from_directory(path_,
                                labels='inferred',
                                label_mode='int',
                                batch_size=train_batch,
                                seed=RANDOM_SEED,
                                shuffle=True,
                                validation_split=0.2,
                                subset='training')
                                
val_set =  image_dataset_from_directory(path_,
                                labels='inferred',
                                label_mode='int',
                                batch_size=test_batch,
                                seed=RANDOM_SEED,
                                shuffle=True,
                                validation_split=0.2,
                                subset='validation')

test_ds_size = int(int(val_set.__len__()) * 0.5) 
test_test_set = val_set.take(test_ds_size)
test_set = val_set.skip(test_ds_size)

vgg19 = VGG19(include_top=False, weights='imagenet', input_shape=(256,256,3))
output = vgg19.layers[-1].output
output = Flatten()(output)
vgg19 = Model(vgg19.input, output)
for layer in vgg19.layers:
    layer.trainable = False
#vgg19.summary()

model_1 = Sequential()

model_1.add(vgg19)
model_1.add(Dense(128,activation='relu', input_dim=(256,256,3), kernel_regularizer=regularizers.L2(0.05)))
model_1.add(Dense(64,activation='relu', kernel_regularizer=regularizers.L2(0.05)))
model_1.add(Dense(1,activation='sigmoid'))

model_1.compile(optimizer='adam',loss='binary_crossentropy',metrics=['accuracy','Recall','Precision','AUC'])

filepath = "Models/TL-vgg19-model-checkpoints/chb-vgg19-model.h5"

callbacks = [ModelCheckpoint(filepath=filepath, monitor="val_accuracy", mode='max', save_best_only=True)]

h = model_1.fit(train_set,
                epochs=100,
                validation_data=test_set,
                callbacks=callbacks
                )

    

model_1.save('chi-vgg19.h5')









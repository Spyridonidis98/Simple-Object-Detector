from keras.models import Model
from keras.layers import Input, Conv2D, MaxPooling2D, Activation, Rescaling, Flatten, Dense, Reshape, BatchNormalization

def SODv2(input_shape, S=15, C=20):
    
    input = Input(input_shape)
    x = Rescaling(1.0/255.0)(input)
    #block 1 size 448
    x = Conv2D(64, 7, strides=(2,2), padding="same")(x)
    x = BatchNormalization()(x) 
    x = Activation("relu")(x)

    #block 2 size 224
    x = Conv2D(64, 3, padding="same")(x)
    x = BatchNormalization()(x) 
    x = Activation("relu")(x)
    x = Conv2D(64, 3, padding="same")(x)
    x = BatchNormalization()(x) 
    x = Activation("relu")(x)
    x = MaxPooling2D((2,2))(x)

    #block 3 size 112
    x = Conv2D(128, 3, padding="same")(x)
    x = BatchNormalization()(x) 
    x = Activation("relu")(x)
    x = Conv2D(128, 3, padding="same")(x)
    x = BatchNormalization()(x) 
    x = Activation("relu")(x)
    x = MaxPooling2D((2,2))(x)

    #block 4 size 56
    x = Conv2D(256, 3, padding="same")(x)
    x = BatchNormalization()(x) 
    x = Activation("relu")(x)
    x = Conv2D(256, 3, padding="same")(x)
    x = BatchNormalization()(x) 
    x = Activation("relu")(x)
    x = Conv2D(256, 3, padding="same")(x)
    x = BatchNormalization()(x) 
    x = Activation("relu")(x)
    x = Conv2D(256, 3, padding="same")(x)
    x = BatchNormalization()(x) 
    x = Activation("relu")(x)
    x = MaxPooling2D((2,2))(x)

    #block 5 size 28 
    x = Conv2D(512, 3, padding="same")(x)
    x = BatchNormalization()(x) 
    x = Activation("relu")(x)
    x = Conv2D(512, 3, padding="same")(x)
    x = BatchNormalization()(x) 
    x = Activation("relu")(x)
    x = Conv2D(512, 3, padding="same")(x)
    x = BatchNormalization()(x) 
    x = Activation("relu")(x)
    x = Conv2D(512, 3, padding="same")(x)
    x = BatchNormalization()(x) 
    x = Activation("relu")(x)
    x = MaxPooling2D((2,2))(x)

    #block 6 size 14
    x = Conv2D(1024, 3, padding="same")(x)
    x = BatchNormalization()(x) 
    x = Activation("relu")(x)
    x = Conv2D(1024, 3, padding="same")(x)
    x = BatchNormalization()(x) 
    x = Activation("relu")(x)
    x = Conv2D(1024, 3, padding="same")(x)
    x = BatchNormalization()(x) 
    x = Activation("relu")(x)
    x = Conv2D(1024, 3, padding="same")(x)
    x = BatchNormalization()(x) 
    x = Activation("relu")(x)
    x = MaxPooling2D((2,2))(x)

    #output size 7
    x = Flatten()(x)
    x = Dense(4096, activation="relu")(x)
    x = BatchNormalization()(x) 
    x = Dense(S*S*(5+C))(x)
    output = Reshape((S,S,5+C), dtype = "float32")(x)

    model = Model(input, output, name="SODv1")
    return model


def cnn_model():
    # Define the network
    image_input = Input(shape=(224, 224, 3))
    pretrained_model = VGG16(weights='imagenet', include_top=False, input_tensor=image_input)

    last_layer = pretrained_model.get_layer('block5_pool').output
    x= Flatten(name='flatten')(last_layer)
    x = Dense(128, activation='relu', name='fc1')(x)
    x = Dense(128, activation='relu', name='fc2')(x)
    out = Dense(1, activation='sigmoid', name='output')(x)
    custom_vgg_model = Model(image_input, out)
    
    for layer in custom_vgg_model.layers[:-3]:
        layer.trainable = False
    
    custom_vgg_model.compile(optimizer=optimizers.SGD(lr=learning_rate), loss='binary_crossentropy', metrics=['accuracy'])
    
    return custom_vgg_model


import tensorflow as tf

AUGMENTATION_CONFIG = {
    'rotation_range': 20,
    'width_shift_range': 0.2,
    'height_shift_range': 0.2,
    'shear_range': 0.15,
    'zoom_range': 0.15,
    'horizontal_flip': True,
    'fill_mode': 'nearest'
}

def get_train_datagen():
    return tf.keras.preprocessing.image.ImageDataGenerator(rescale=1./255, **AUGMENTATION_CONFIG)

def get_val_datagen():
    return tf.keras.preprocessing.image.ImageDataGenerator(rescale=1./255)
AUGMENTATION_CONFIG = {
    'rotation_range': 20,
    'width_shift_range': 0.2,
    'height_shift_range': 0.2,
    'shear_range': 0.15,
    'zoom_range': 0.15,
    'horizontal_flip': True,
    'fill_mode': 'nearest'
}

def get_train_datagen():
    return tf.keras.preprocessing.image.ImageDataGenerator(rescale=1./255, **AUGMENTATION_CONFIG)

def get_val_datagen():
    return tf.keras.preprocessing.image.ImageDataGenerator(rescale=1./255)

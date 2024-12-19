# Importing the Keras libraries and packages
from keras.models import Sequential
from keras.layers import Convolution2D, MaxPooling2D, Flatten, Dense, Dropout
from keras.optimizers import Adam
from keras.preprocessing.image import ImageDataGenerator

# Define input image dimensions and number of classes
sz = 128
num_classes = 3  # 'D', 'R', 'U'

# Step 1 - Building the CNN architecture for 'DRU'
classifier_dru = Sequential()

# First convolution layer and pooling
classifier_dru.add(Convolution2D(32, (3, 3), input_shape=(sz, sz, 1), activation='relu'))
classifier_dru.add(MaxPooling2D(pool_size=(2, 2)))

# Second convolution layer and pooling
classifier_dru.add(Convolution2D(32, (3, 3), activation='relu'))
classifier_dru.add(MaxPooling2D(pool_size=(2, 2)))

# Flattening the layers
classifier_dru.add(Flatten())

# Adding fully connected layers with dropout for regularization
classifier_dru.add(Dense(units=128, activation='relu'))
classifier_dru.add(Dropout(0.4))
classifier_dru.add(Dense(units=num_classes, activation='softmax'))

# Compiling the CNN
classifier_dru.compile(optimizer=Adam(), loss='categorical_crossentropy', metrics=['accuracy'])

# Step 2 - Preparing the train/test data for 'DRU'
train_datagen_dru = ImageDataGenerator(
    rescale=1./255,
    shear_range=0.2,
    zoom_range=0.2,
    horizontal_flip=True)

test_datagen_dru = ImageDataGenerator(rescale=1./255)

training_set_dru = train_datagen_dru.flow_from_directory(
    'data/train_dru',
    target_size=(sz, sz),
    batch_size=10,
    color_mode='grayscale',
    class_mode='categorical')

test_set_dru = test_datagen_dru.flow_from_directory(
    'data/test_dru',
    target_size=(sz, sz),
    batch_size=10,
    color_mode='grayscale',
    class_mode='categorical')

# Training the 'DRU' model
classifier_dru.fit_generator(
    training_set_dru,
    steps_per_epoch=280,
    epochs=10,
    validation_data=test_set_dru,
    validation_steps=102)

# Saving the 'DRU' model
model_json_dru = classifier_dru.to_json()
with open("model-bw_dru.json", "w") as json_file:
    json_file.write(model_json_dru)
classifier_dru.save_weights('model-bw_dru.h5')
print('DRU Model Saved')

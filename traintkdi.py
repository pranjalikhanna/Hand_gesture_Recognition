# Importing the Keras libraries and packages
from keras.models import Sequential
from keras.layers import Convolution2D, MaxPooling2D, Flatten, Dense, Dropout
from keras.optimizers import Adam
from keras.preprocessing.image import ImageDataGenerator

# Define input image dimensions and number of classes
sz = 128
num_classes = 4  # 'T', 'K', 'D','I'

# Step 1 - Building the CNN architecture for 'tkdi'
classifier_tkdi = Sequential()

# First convolution layer and pooling
classifier_tkdi.add(Convolution2D(32, (3, 3), input_shape=(sz, sz, 1), activation='relu'))
classifier_tkdi.add(MaxPooling2D(pool_size=(2, 2)))

# Second convolution layer and pooling
classifier_tkdi.add(Convolution2D(32, (3, 3), activation='relu'))
classifier_tkdi.add(MaxPooling2D(pool_size=(2, 2)))

# Flattening the layers
classifier_tkdi.add(Flatten())

# Adding fully connected layers with dropout for regularization
classifier_tkdi.add(Dense(units=128, activation='relu'))
classifier_tkdi.add(Dropout(0.4))
classifier_tkdi.add(Dense(units=num_classes, activation='softmax'))

# Compiling the CNN
classifier_tkdi.compile(optimizer=Adam(), loss='categorical_crossentropy', metrics=['accuracy'])

# Step 2 - Preparing the train/test data for 'tkdi'
train_datagen_tkdi = ImageDataGenerator(
    rescale=1./255,
    shear_range=0.2,
    zoom_range=0.2,
    horizontal_flip=True)

test_datagen_tkdi = ImageDataGenerator(rescale=1./255)

training_set_tkdi = train_datagen_tkdi.flow_from_directory(
    'data/train_tkdi',
    target_size=(sz, sz),
    batch_size=10,
    color_mode='grayscale',
    class_mode='categorical')

test_set_tkdi = test_datagen_tkdi.flow_from_directory(
    'data/test_tkdi',
    target_size=(sz, sz),
    batch_size=10,
    color_mode='grayscale',
    class_mode='categorical')

# Training the 'tkdi' model
classifier_tkdi.fit_generator(
    training_set_tkdi,
    steps_per_epoch=367,
    epochs=10,
    validation_data=test_set_tkdi,
    validation_steps=122)

# Saving the 'tkdi' model
model_json_tkdi = classifier_tkdi.to_json()
with open("model-bw_tkdi.json", "w") as json_file:
    json_file.write(model_json_tkdi)
classifier_tkdi.save_weights('model-bw_tkdi.h5')
print('tkdi Model Saved')

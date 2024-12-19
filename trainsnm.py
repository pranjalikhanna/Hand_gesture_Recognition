# Importing the Keras libraries and packages
from keras.models import Sequential
from keras.layers import Convolution2D, MaxPooling2D, Flatten, Dense, Dropout
from keras.optimizers import Adam
from keras.preprocessing.image import ImageDataGenerator

# Define input image dimensions and number of classes
sz = 128
num_classes = 3  # 's', 'N', 'M'

# Step 1 - Building the CNN architecture for 'snm'
classifier_snm = Sequential()

# First convolution layer and pooling
classifier_snm.add(Convolution2D(32, (3, 3), input_shape=(sz, sz, 1), activation='relu'))
classifier_snm.add(MaxPooling2D(pool_size=(2, 2)))

# Second convolution layer and pooling
classifier_snm.add(Convolution2D(32, (3, 3), activation='relu'))
classifier_snm.add(MaxPooling2D(pool_size=(2, 2)))

# Flattening the layers
classifier_snm.add(Flatten())

# Adding fully connected layers with dropout for regularization
classifier_snm.add(Dense(units=128, activation='relu'))
classifier_snm.add(Dropout(0.4))
classifier_snm.add(Dense(units=num_classes, activation='softmax'))

# Compiling the CNN
classifier_snm.compile(optimizer=Adam(), loss='categorical_crossentropy', metrics=['accuracy'])

# Step 2 - Preparing the train/test data for 'snm'
train_datagen_snm = ImageDataGenerator(
    rescale=1./255,
    shear_range=0.2,
    zoom_range=0.2,
    horizontal_flip=True)

test_datagen_snm = ImageDataGenerator(rescale=1./255)

training_set_snm = train_datagen_snm.flow_from_directory(
    'data/train_snm',
    target_size=(sz, sz),
    batch_size=10,
    color_mode='grayscale',
    class_mode='categorical')

test_set_snm = test_datagen_snm.flow_from_directory(
    'data/test_snm',
    target_size=(sz, sz),
    batch_size=10,
    color_mode='grayscale',
    class_mode='categorical')

# Training the 'SNM' model
classifier_snm.fit_generator(
    training_set_snm,
    steps_per_epoch=308,
    epochs=10,
    validation_data=test_set_snm,
    validation_steps=98)

# Saving the 'snm' model
model_json_snm = classifier_snm.to_json()
with open("model-bw_snm.json", "w") as json_file:
    json_file.write(model_json_snm)
classifier_snm.save_weights('model-bw_snm.h5')
print('snm Model Saved')

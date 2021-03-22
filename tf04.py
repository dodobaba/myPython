from tensorflow import keras
from tensorflow.keras.preprocessing.image import ImageDataGenerator
from tensorflow.keras.optimizers import RMSprop


train_image = ImageDataGenerator(rescale=1/255)
validation_image = ImageDataGenerator(rescale=1/255)

train_generator = train_image.flow_from_directory(
    'horse-or-human',
    target_size=(300, 300),
    batch_size=32,
    class_mode='binary')

validation_generator = validation_image.flow_from_directory(
    'validation-horse-or-human',
    target_size=(300, 300),
    batch_size=32,
    class_mode='binary'
)

model = keras.Sequential([
    keras.layers.Conv2D(16, (3, 3), activation='relu', input_shape=(300, 300, 3)),
    keras.layers.MaxPool2D(2, 2),
    keras.layers.Conv2D(32, (3, 3), activation='relu'),
    keras.layers.MaxPool2D(2, 2),
    keras.layers.Conv2D(64, (3, 3), activation='relu'),
    keras.layers.MaxPool2D(2, 2),
    keras.layers.Flatten(),
    keras.layers.Dense(512, activation='relu'),
    keras.layers.Dense(1, activation='sigmoid')
])

model.compile(
    loss='binary_crossentropy',
    optimizer=RMSprop(lr=0.001),
    metrics=['acc']
)

model.summary()

history = model.fit(
    train_generator,
    epochs=5,
    verbose=1,
    validation_data=validation_generator,
    validation_steps=8
)




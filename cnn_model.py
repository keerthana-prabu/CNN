
# 1. IMPORT LIBRARIES
import os
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns

from PIL import Image

import tensorflow as tf
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import (
    Conv2D,
    MaxPooling2D,
    Flatten,
    Dense,
    Dropout
)

from tensorflow.keras.preprocessing.image import ImageDataGenerator

from tensorflow.keras.callbacks import (
    EarlyStopping,
    ReduceLROnPlateau
)

from sklearn.metrics import (
    classification_report,
    confusion_matrix
)

import gradio as gr


train_dir = "dataset/intel-image-classification/seg_train/seg_train"

test_dir = "dataset/intel-image-classification/seg_test/seg_test"


classes = sorted(os.listdir(train_dir))

print("DATASET INFORMATION")

print("Classes:", classes)
print("Number of classes:", len(classes))


# Count images in each class

class_counts = {}

for class_name in classes:

    class_path = os.path.join(train_dir, class_name)

    image_count = len(os.listdir(class_path))

    class_counts[class_name] = image_count

    print(f"{class_name}: {image_count}")


print("Total training images:", sum(class_counts.values()))

# 4. EDA - CLASS DISTRIBUTION
plt.figure(figsize=(10, 6))

plt.bar(
    class_counts.keys(),
    class_counts.values()
)

plt.xlabel("Class")
plt.ylabel("Number of Images")
plt.title("Training Dataset - Class Distribution")

plt.xticks(rotation=45)

plt.tight_layout()

plt.show()


# 5. EDA - DISPLAY SAMPLE IMAGE FROM EACH CLASS
plt.figure(figsize=(12, 8))

for i, class_name in enumerate(classes):

    class_path = os.path.join(train_dir, class_name)

    image_name = os.listdir(class_path)[0]

    image_path = os.path.join(
        class_path,
        image_name
    )

    image = plt.imread(image_path)

    plt.subplot(2, 3, i + 1)

    plt.imshow(image)

    plt.title(class_name)

    plt.axis("off")


plt.tight_layout()

plt.show()


# 6. EDA - CHECK IMAGE DIMENSIONS
sample_class = classes[0]

sample_class_path = os.path.join(
    train_dir,
    sample_class
)

sample_image_name = os.listdir(
    sample_class_path
)[0]

sample_image_path = os.path.join(
    sample_class_path,
    sample_image_name
)

sample_image = Image.open(sample_image_path)

print("IMAGE INFORMATION")

print("Sample image:", sample_image_name)

print("Image size:", sample_image.size)

print("Image mode:", sample_image.mode)


# 7. IMAGE CONFIGURATION
IMG_SIZE = (150, 150)

BATCH_SIZE = 32

EPOCHS = 10


# 8. DATA PREPROCESSING + DATA AUGMENTATION

train_datagen = ImageDataGenerator(

    rescale=1.0 / 255,
    validation_split=0.2,

    # Data augmentation

    rotation_range=20,
    width_shift_range=0.1,
    height_shift_range=0.1,
    zoom_range=0.2,
    horizontal_flip=True
)


validation_datagen = ImageDataGenerator(

    rescale=1.0 / 255,

    validation_split=0.2
)


# 9. CREATE TRAINING DATA GENERATOR

train_generator = train_datagen.flow_from_directory(

    train_dir,

    target_size=IMG_SIZE,

    batch_size=BATCH_SIZE,

    class_mode="categorical",

    subset="training",

    shuffle=True
)


# 10. CREATE VALIDATION DATA GENERATOR

validation_generator = validation_datagen.flow_from_directory(

    train_dir,

    target_size=IMG_SIZE,

    batch_size=BATCH_SIZE,

    class_mode="categorical",

    subset="validation",

    shuffle=False
)


# 11. DISPLAY CLASS MAPPING

print("CLASS MAPPING")

print(train_generator.class_indices)


# 12. CREATE TEST DATA GENERATOR

test_datagen = ImageDataGenerator(

    rescale=1.0 / 255
)


test_generator = test_datagen.flow_from_directory(

    test_dir,

    target_size=IMG_SIZE,

    batch_size=BATCH_SIZE,

    class_mode="categorical",

    shuffle=False
)


# 13. BUILD CNN MODEL

model = Sequential()


# First Convolution Block
model.add(

    Conv2D(

        filters=32,

        kernel_size=(3, 3),

        activation="relu",

        input_shape=(150, 150, 3)
    )
)


model.add(

    MaxPooling2D(

        pool_size=(2, 2)
    )
)


# Second Convolution Block
model.add(

    Conv2D(

        filters=64,

        kernel_size=(3, 3),

        activation="relu"
    )
)


model.add(

    MaxPooling2D(

        pool_size=(2, 2)
    )
)


# Third Convolution Block
model.add(

    Conv2D(

        filters=128,

        kernel_size=(3, 3),

        activation="relu"
    )
)


model.add(

    MaxPooling2D(

        pool_size=(2, 2)
    )
)


# 14. CONVERT CNN OUTPUT INTO A VECTOR

model.add(

    Flatten()
)


# 15. FULLY CONNECTED LAYER

model.add(

    Dense(

        128,

        activation="relu"
    )
)


# 16. DROPOUT TO REDUCE OVERFITTING

model.add(
    Dropout(0.5)
)


# 17. OUTPUT LAYER

model.add(
    Dense(
        6,
        activation="softmax"
    )
)


# 18. DISPLAY MODEL ARCHITECTURE

print("CNN MODEL SUMMARY")

model.summary()


# 19. COMPILE MODEL

model.compile(

    optimizer="adam",

    loss="categorical_crossentropy",

    metrics=["accuracy"]
)


# 20. CALLBACKS


early_stopping = EarlyStopping(

    monitor="val_loss",

    patience=3,

    restore_best_weights=True
)


reduce_lr = ReduceLROnPlateau(

    monitor="val_loss",

    factor=0.2,

    patience=2,

    min_lr=0.00001
)


# 21. TRAIN CNN

print("TRAINING CNN")

history = model.fit(

    train_generator,

    validation_data=validation_generator,

    epochs=EPOCHS,

    callbacks=[
        early_stopping,
        reduce_lr
    ]
)


# 22. PLOT TRAINING AND VALIDATION ACCURACY

plt.figure(figsize=(10, 6))

plt.plot(

    history.history["accuracy"],

    label="Training Accuracy"
)

plt.plot(

    history.history["val_accuracy"],

    label="Validation Accuracy"
)

plt.xlabel("Epoch")

plt.ylabel("Accuracy")

plt.title("Training vs Validation Accuracy")

plt.legend()

plt.show()


# 23. PLOT TRAINING AND VALIDATION LOSS

plt.figure(figsize=(10, 6))

plt.plot(

    history.history["loss"],

    label="Training Loss"
)

plt.plot(

    history.history["val_loss"],

    label="Validation Loss"
)

plt.xlabel("Epoch")

plt.ylabel("Loss")

plt.title("Training vs Validation Loss")

plt.legend()

plt.show()


# 24. EVALUATE MODEL ON TEST DATA

print("TEST SET EVALUATION")


test_loss, test_accuracy = model.evaluate(

    test_generator
)


print("Test Loss:", test_loss)

print("Test Accuracy:", test_accuracy)


# 25. GENERATE PREDICTIONS

predictions = model.predict(

    test_generator
)


# Convert probabilities to class numbers

y_pred = np.argmax(

    predictions,

    axis=1
)


# Actual class numbers

y_true = test_generator.classes


# 26. CLASSIFICATION REPORT

class_names = list(

    test_generator.class_indices.keys()

)


print("CLASSIFICATION REPORT")

print(

    classification_report(

        y_true,

        y_pred,

        target_names=class_names
    )

)


# 27. CONFUSION MATRIX
cm = confusion_matrix(

    y_true,

    y_pred
)


plt.figure(figsize=(9, 7))


sns.heatmap(

    cm,

    annot=True,

    fmt="d",

    xticklabels=class_names,

    yticklabels=class_names
)


plt.xlabel("Predicted Class")

plt.ylabel("Actual Class")

plt.title("Confusion Matrix")

plt.tight_layout()

plt.show()


# 28. SAVE TRAINED MODEL

model.save(

    "intel_scene_cnn.keras"
)


print("\nModel saved as: intel_scene_cnn.keras")


# 29. GRADIO PREDICTION FUNCTION

def predict_image(image):

    # Resize image

    image = image.resize(IMG_SIZE)


    # Convert image to NumPy array

    image_array = np.array(image)


    # Normalize pixel values

    image_array = image_array / 255.0


    # Add batch dimension

    image_array = np.expand_dims(

        image_array,

        axis=0
    )


    # Make prediction

    prediction = model.predict(

        image_array,

        verbose=0
    )[0]


    # Create dictionary containing
    # class probabilities

    results = {}

    for i in range(len(class_names)):

        results[class_names[i]] = float(

            prediction[i]

        )


    return results


# 30. CREATE GRADIO APPLICATION

demo = gr.Interface(

    fn=predict_image,

    inputs=gr.Image(

        type="pil"
    ),

    outputs=gr.Label(

        num_top_classes=6
    ),

    title="Intel Scene Classification using CNN",

    description=(
        "Upload an image and the CNN will "
        "predict whether it is a building, "
        "forest, glacier, mountain, sea, or street."
    )
)


# 31. LAUNCH APPLICATION
demo.launch()
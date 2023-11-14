import matplotlib.pyplot as plt
import numpy as np
import PIL
import tensorflow as tf
from tensorflow import keras
from keras import layers
from keras.models import Sequential
from keras.preprocessing import image
import pathlib
import matplotlib.pyplot as plt
from keras.callbacks import EarlyStopping
from keras.utils import img_to_array, load_img
from sklearn.metrics import precision_recall_curve, confusion_matrix
from sklearn.preprocessing import label_binarize
import seaborn as sns
import os
from shutil import copyfile

# load dataset
dataset = pathlib.Path("dataset/train")

# Make directories that we will use for saving failure cases and trained models
misclassified_dir = "misclassified_images"
os.makedirs(misclassified_dir, exist_ok=True)

model_save_directory = "trained_models"
os.makedirs(model_save_directory, exist_ok=True)

image_count = len(list(dataset.glob('*/*.png')))
print(image_count)

batch_size = 1000
img_height = 180
img_width = 180

train_ds = tf.keras.utils.image_dataset_from_directory(
  dataset,
  validation_split=0.2,
  subset="training",
  seed=123,
  image_size=(img_height, img_width),
  batch_size=batch_size
)

val_ds = tf.keras.utils.image_dataset_from_directory(
  dataset,
  validation_split=0.2,
  subset="validation",
  seed=123,
  image_size=(img_height, img_width),
  batch_size=batch_size
)

class_names = train_ds.class_names
print(class_names)

for image_batch, labels_batch in train_ds:
    print(image_batch.shape)
    print(labels_batch.shape)
    break

AUTOTUNE = tf.data.AUTOTUNE

train_ds = train_ds.cache().shuffle(1000).prefetch(buffer_size=AUTOTUNE)
val_ds = val_ds.cache().prefetch(buffer_size=AUTOTUNE)

normalization_layer = layers.Rescaling(1./255)

normalized_ds = train_ds.map(lambda x, y: (normalization_layer(x), y))
image_batch, labels_batch = next(iter(normalized_ds))
first_image = image_batch[0]
# Notice the pixel values are now in `[0,1]`.
print(np.min(first_image), np.max(first_image))

num_classes = len(class_names)

# Model training
model = Sequential([
  layers.Rescaling(1./255, input_shape=(img_height, img_width, 3)),
  layers.Conv2D(16, 3, padding='same', activation='relu'),
  layers.MaxPooling2D(),
  layers.Conv2D(32, 3, padding='same', activation='relu'),
  layers.MaxPooling2D(),
  layers.Conv2D(64, 3, padding='same', activation='relu'),
  layers.MaxPooling2D(),
  # layers.Dropout(0.2),
  layers.Flatten(),
  layers.Dense(128, activation='relu'),
  layers.Dense(num_classes)
])

model.compile(optimizer='adam',
              loss=tf.keras.losses.SparseCategoricalCrossentropy(from_logits=True),
              metrics=['accuracy'])

model.summary()

# Dont worry about early stopping for now. It doesnt work anyway :kekw:
early_stopping = EarlyStopping(
    monitor='val_loss',  
    patience=3,           
    restore_best_weights=True  
)

epochs=25

history = model.fit(
  train_ds,
  validation_data=val_ds,
  epochs=epochs,
  # callbacks=[early_stopping]
)

model.save("trained_models/emotional_cnn")

acc = history.history['accuracy']
val_acc = history.history['val_accuracy']

loss = history.history['loss']
val_loss = history.history['val_loss']

epochs_range = range(epochs)


# Plotting loss graphs and accuracy graphs to check for overfitting and other analysis stuff
plt.figure(figsize=(8, 8))
plt.subplot(1, 2, 1)
plt.plot(epochs_range, acc, label='Training Accuracy')
plt.plot(epochs_range, val_acc, label='Validation Accuracy')
plt.legend(loc='lower right')
plt.title('Training and Validation Accuracy')

plt.subplot(1, 2, 2)
plt.plot(epochs_range, loss, label='Training Loss')
plt.plot(epochs_range, val_loss, label='Validation Loss')
plt.legend(loc='upper right')
plt.title('Training and Validation Loss')
plt.show()

true_labels = []
predicted_probabilities = []

prediction = keras.models.load_model("trained_models/emotional_cnn")

test_dataset = pathlib.Path("dataset/test")

class_names = sorted([item.name for item in test_dataset.glob('*')])

# This is basically using the trained model to test images it has never seen before and predict its class, and also save any misclassified cases
for class_name in class_names:
  class_path = test_dataset / class_name
  # i = 0
  for img_path in class_path.glob('*.png'):
    img = image.load_img(img_path, target_size=(180, 180))
    img_array = image.img_to_array(img)
    img_array = np.expand_dims(img_array, axis=0)

    predictions = prediction.predict(img_array)
    score = tf.nn.softmax(predictions[0])

    true_label = class_names.index(class_name)
    true_labels.append(true_label)
    predicted_probabilities.append(score)

    predicted_label = np.argmax(predictions[0])

    print(f"Image: {img_path}, Predicted Class: {class_names[np.argmax(score)]}, Confidence: {100 * np.max(score):.2f}%")

    if predicted_label != true_label:
      # Save the misclassified image
      misclassified_image_path = os.path.join(misclassified_dir, f"true_{class_names[true_label]}_pred_{class_names[predicted_label]}_{os.path.basename(img_path)}")
      copyfile(img_path, misclassified_image_path)

      print(f"Image {os.path.basename(img_path)} misclassified. True Label: {class_names[true_label]}, Predicted Label: {class_names[predicted_label]}")

    # i += 1

    # if (i > 15):
    #   break

true_labels = np.array(true_labels)
predicted_probabilities = np.array(predicted_probabilities).squeeze()
binarized_labels = label_binarize(true_labels, classes=np.arange(len(class_names)))
predicted_labels = np.argmax(predicted_probabilities, axis=1)

precision = dict()
recall = dict()
for i in range(len(class_names)):
  precision[i], recall[i], _ = precision_recall_curve(binarized_labels[:, i], predicted_probabilities[:, i])
precision["micro"], recall["micro"], _ = precision_recall_curve(binarized_labels.ravel(), predicted_probabilities.ravel())

# Precision Recall Curve
plt.figure(figsize=(8, 8))
plt.plot(recall["micro"], precision["micro"], label=f'Precision-Recall Curve (Micro-average)')
plt.xlabel('Recall')
plt.ylabel('Precision')
plt.title('Precision-Recall Curve (Micro-average)')
plt.legend()
plt.show()

# Confusion Matrix
conf_matrix = confusion_matrix(true_labels, predicted_labels)
plt.figure(figsize=(10, 8))
sns.heatmap(conf_matrix, annot=True, fmt="d", cmap="Blues", xticklabels=class_names, yticklabels=class_names)
plt.xlabel('Predicted Labels')
plt.ylabel('True Labels')
plt.title('Confusion Matrix')
plt.show()
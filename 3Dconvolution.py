import tensorflow as tf
from tensorflow.keras import layers, models

model = models.Sequential()
model.add(layers.Conv3D(16, (3, 3, 3), activation='relu', input_shape=(10, 64, 64, 1)))

input_tensor = tf.random.normal([1, 10, 64, 64, 1])

output_tensor = model(input_tensor)

print(output_tensor.shape)
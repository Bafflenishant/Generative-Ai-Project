import numpy as np
import tensorflow as tf

generator = tf.keras.models.load_model("D:\AI_Projects\GEN-NY\GEN-NY\models\generator_model.h5")

noise_dim = 100
random_noise = np.random.randn(1, noise_dim)

print("Predicting...")
generated_image = generator.predict(random_noise)
print("Prediction successful!")

print("Generated Image Shape:", generated_image.shape)
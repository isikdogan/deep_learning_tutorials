import tensorflow as tf

(train_images, train_labels), (test_images, test_labels) = tf.keras.datasets.mnist.load_data()

inputs = tf.keras.Input(shape=(28,28))
net = tf.keras.layers.Flatten()(inputs)
net = tf.keras.layers.Dense(512, activation=tf.nn.relu)(net)
net = tf.keras.layers.Dense(10, activation=tf.nn.softmax)(net)
model = tf.keras.Model(inputs=inputs, outputs=net)

model.compile(optimizer='adam', 
              loss='sparse_categorical_crossentropy',
              metrics=['accuracy'])

model.fit(train_images, train_labels, epochs=5)

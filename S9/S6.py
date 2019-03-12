import tensorflow as tf

data_train, _ = tf.keras.datasets.mnist.load_data()
dataset = tf.data.Dataset.from_tensor_slices(data_train)
dataset = dataset.shuffle(buffer_size=60000)
dataset = dataset.batch(32)

strategy = tf.distribute.MirroredStrategy()
with strategy.scope():
    model = tf.keras.Sequential([
        tf.keras.layers.Flatten(input_shape=(28, 28)),
        tf.keras.layers.Dense(512, activation=tf.nn.relu),
        tf.keras.layers.Dense(10, activation=tf.nn.softmax)
    ])

    model.compile(optimizer='adam', 
                  loss='sparse_categorical_crossentropy',
                  metrics=['accuracy'])

    tb = tf.keras.callbacks.TensorBoard(log_dir='./checkpoints')
    model.fit(dataset, epochs=5, callbacks=[tb])

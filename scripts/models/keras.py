import tensorflow as tf
import numpy
from tensorflow import keras
import numpy as np

input = tf.placeholder(tf.string, None)

country = tf.feature_column.indicator_column(tf.feature_column.categorical_column_with_vocabulary_list('co', ['US', 'FI']))

columns = [country]

features = tf.parse_example(serialized=input, features=tf.feature_column.make_parse_example_spec(columns))
inputvec = tf.feature_column.input_layer(features, columns)

# Instantiate the model given inputs and outputs.
model = keras.Sequential()
inputLayer = keras.layers.InputLayer(input_shape=(2,), input_tensor=inputvec)
inputLayer.is_placeholder = True
model.add(inputLayer)
model.add(keras.layers.Dense(2, activation='softmax'))

# The compile step specifies the training configuration.
model.compile(optimizer=tf.train.AdamOptimizer(0.001),
              loss='categorical_crossentropy',
              metrics=['accuracy'])


model.fit(x=np.matrix([[0, 1], [1, 0]]), y=keras.utils.to_categorical(np.array([0, 1]), num_classes=2), epochs=10, batch_size=32)

sess = tf.Session()

sess.run(tf.global_variables_initializer())

res = sess.run(model.output, feed_dict={inputvec: [[0, 1]]})
print(res)

legacy_init_op = tf.group(tf.tables_initializer(), name='legacy_init_op')

tf.saved_model.simple_save(
    sess,
    '../../testdata/test_models/keras/1/',
    {
        'input': input,
    },
    {
        'output': model.output[0][0] > 0.5,
    },
    legacy_init_op
)

# save test file
import json
with open('keras_out.json', 'w') as outfile:
    json.dump({"output": res.tolist()[0][0] > 0.5}, outfile)
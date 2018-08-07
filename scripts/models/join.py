import tensorflow as tf

a = tf.placeholder(tf.string, None)
b = tf.placeholder(tf.string, None)

c = tf.string_join([a, b])

sess = tf.Session()

tf.saved_model.simple_save(
    sess,
    '../../testdata/test_models/join/1/',
    {'a': a, 'b': b},
    {'joined': c}
)
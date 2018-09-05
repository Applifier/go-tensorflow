import tensorflow as tf

# This is used to verify that both SavedModel predictor and TF serving return values the same way

single = tf.placeholder(tf.int64, shape=(1))
vector = tf.placeholder(tf.int64)
matrix = tf.placeholder(tf.int64, shape=(3, 2))

sess = tf.Session()
tf.saved_model.simple_save(
    sess,
    '../testdata/models/test/1/',
    {
        'single': single,
        'vector': vector,
        'matrix': matrix,
    },
    {   'input_single': tf.identity(single),
        'input_vector': tf.identity(vector),
        'input_matrix': tf.identity(matrix),
        'single': tf.constant(1),
        'vector': tf.constant([1,2,3]),
        'matrix': tf.constant([[1, 2], [3, 4], [5, 6]]),
    }
)

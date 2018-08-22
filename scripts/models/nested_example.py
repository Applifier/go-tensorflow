import tensorflow as tf

input = tf.placeholder(tf.string, None)

'''
{
    "input": {
        "foo": {
            "bar": "bar"
        }
    }
}
'''

root = tf.parse_single_example(input[0], features={
    'foo': tf.FixedLenFeature(shape=[], dtype=tf.string),
})

foo = tf.parse_single_example(root['foo'], features={
    'bar': tf.FixedLenFeature(shape=[], dtype=tf.string),
})

sess = tf.Session()

tf.saved_model.simple_save(
    sess,
    '../../testdata/test_models/nested_example/1/',
    {
        'input': input
    },
    {
        'foo': foo['bar'],
    }
)
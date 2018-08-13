import tensorflow as tf

# Helper function for creating hash tables from pydict
def createMap(d, default):
    table = tf.contrib.lookup.HashTable(
        tf.contrib.lookup.KeyValueTensorInitializer(list(d.keys()), list(d.values())), default)
    def lookup(key):
        return table.lookup(key)

    return lookup


input = tf.placeholder(tf.string, None)
features = tf.parse_example(
        input,
        # Defaults are not specified since both keys are required.
        features={
            'co': tf.FixedLenFeature([], tf.string),
        })


countryMap = createMap({"FI": "Finland", "US": "United States of America"}, "unknown")
langMap = createMap({"FI": "Finnish", "US": "English"}, "unknown")

coName = countryMap(features['co'])
lang =  langMap(features['co'])

sess = tf.Session()
sess.run(tf.tables_initializer())

legacy_init_op = tf.group(tf.tables_initializer(), name='legacy_init_op')

tf.saved_model.simple_save(
    sess,
    '../../testdata/test_models/table/1/',
    {'input': input},
    {   
        'name': coName,
        'lang': lang,
    },
    legacy_init_op
)
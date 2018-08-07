import tensorflow as tf

input = tf.placeholder(tf.string, None)

# First, convert the raw input to a numeric column.
numeric_feature_column = tf.feature_column.numeric_column("year")

# Then, bucketize the numeric column on the years 1960, 1980, and 2000.
bucketized_feature_column = tf.feature_column.bucketized_column(
    source_column = numeric_feature_column,
    boundaries = [1960, 1980, 2000])

columns = [numeric_feature_column, bucketized_feature_column]

features = tf.parse_example(serialized=input, features=tf.feature_column.make_parse_example_spec(columns))

dense_tensor = tf.feature_column.input_layer(features, columns)

sess = tf.Session()

tf.saved_model.simple_save(
    sess,
    '../../testdata/test_models/feature_column/1/',
    {'input': input},
    {   
        'year': features['year'], 
        'dense': dense_tensor
    }
)
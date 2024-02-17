"""Transform module
"""
 
import tensorflow as tf
import tensorflow_transform as tft

# change here
CATEGORICAL_FEATURES = {
    "localization": 15,
    "sex": 3
}
NUMERICAL_FEATURES = [
    "age",
]
LABEL_KEY = "dx"
 
def transformed_name(key):
    """Renaming transformed features"""
    return key + "_xf"
 
def convert_num_to_one_hot(label_tensor, num_labels=2):
    """
    Convert a label (0 or 1) into a one-hot vector
    Args:
        int: label_tensor (0 or 1)
    Returns
        label tensor
    """
    one_hot_tensor = tf.one_hot(label_tensor, num_labels)
    return tf.reshape(one_hot_tensor, [-1, num_labels])
 
 
def preprocessing_fn(inputs):
    """
    Preprocess input features into transformed features
    
    Args:
        inputs: map from feature keys to raw features.
    
    Return:
        outputs: map from feature keys to transformed features.    
    """
    
    outputs = {}
    
    for key in CATEGORICAL_FEATURES:
        dim = CATEGORICAL_FEATURES[key]
        int_value = tft.compute_and_apply_vocabulary(
            inputs[key], top_k=dim + 1
        )
        outputs[transformed_name(key)] = convert_num_to_one_hot(
            int_value, num_labels=dim + 1
        )
    
    for feature in NUMERICAL_FEATURES:
        outputs[transformed_name(feature)] = tft.scale_to_0_1(inputs[feature])
    
    
    # outputs[transformed_name(LABEL_KEY)] = tf.cast(inputs[LABEL_KEY], tf.string)
    # For the label column we provide the mapping from string to index.
    table_keys = ["nv", "mel", "bkl", "bcc", "akiec", "df", "vasc"]
    with tf.init_scope():
        initializer = tf.lookup.KeyValueTensorInitializer(
            keys=table_keys,
            values=tf.cast(tf.range(len(table_keys)), tf.int64),
            key_dtype=tf.string,
            value_dtype=tf.int64)
        table = tf.lookup.StaticHashTable(initializer, default_value=-1)

    # Remove trailing periods for test data when the data is read with tf.data.
    # label_str  = tf.sparse.to_dense(inputs[LABEL_KEY])
    label_str = inputs[LABEL_KEY]
    label_str = tf.strings.lower(label_str)
    data_labels = table.lookup(label_str)
    transformed_label = tf.one_hot(
          indices=data_labels, depth=len(table_keys), on_value=1.0, off_value=0.0)
    outputs[transformed_name(LABEL_KEY)] = tf.reshape(transformed_label, [-1, len(table_keys)])
    
    return outputs
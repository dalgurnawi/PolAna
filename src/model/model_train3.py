import tensorflow as tf
import tensorflow_hub as hub
import tensorflow_text as text


# Function for preprocessing
# (will probably be part of tensorflow-text soon)
def load_vocab(vocab_file):
    """Loads a vocabulary file into a list."""
    vocab = []
    with tf.io.gfile.GFile(vocab_file, "r") as reader:
        while True:
            token = reader.readline()
            if not token:
                break
            token = token.strip()
            vocab.append(token)
    return vocab


def create_vocab_table(vocab, num_oov=1):
    vocab_values = tf.range(tf.size(vocab, out_type=tf.int64), dtype=tf.int64)
    init = tf.lookup.KeyValueTensorInitializer(keys=vocab, values=vocab_values, key_dtype=tf.string, value_dtype=tf.int64)
    vocab_table = tf.lookup.StaticVocabularyTable(init, num_oov, lookup_key_dtype=tf.string)
    return vocab_table


def merge_dims(rt, axis=0):
    """Collapses the specified axis of a RaggedTensor.

    Suppose we have a RaggedTensor like this:
    [[1, 2, 3],
    [4, 5],
    [6]]

    If we flatten the 0th dimension, it becomes:
    [1, 2, 3, 4, 5, 6]

    Args:
    rt: a RaggedTensor.
    axis: the dimension to flatten.

    Returns:
    A flattened RaggedTensor, which now has one less dimension.
    """
    to_expand = rt.nested_row_lengths()[axis]
    to_elim = rt.nested_row_lengths()[axis + 1]

    bar = tf.RaggedTensor.from_row_lengths(to_elim, row_lengths=to_expand)
    new_row_lengths = tf.reduce_sum(bar, axis=1)
    return tf.RaggedTensor.from_nested_row_lengths(rt.flat_values, rt.nested_row_lengths()[:axis] + (new_row_lengths,))

# Load BERT model from TF Hub
model_url = "https://tfhub.dev/tensorflow/bert_en_cased_L-12_H-768_A-12/1"
bert_layer = hub.KerasLayer(model_url, trainable=False)

# Load vocabulary from BERT TF Hub model
vocab_file = bert_layer.resolved_object.vocab_file.asset_path.numpy()
vocab = load_vocab(vocab_file)

# vocab_file = layer.resolved_object.vocab_file.asset_path # here is the change
# tokenizer = text.BertTokenizer(vocab_file)
# tokens = tokenizer.tokenize(text_inputs)

# Check if BERT model is case sensitive
do_lower_case = bert_layer.resolved_object.do_lower_case.numpy()

num_oov = 1
# Create the vocabulary table
vocab_table = create_vocab_table(vocab, num_oov)

# Define some inputs and tokenize it
text_inputs = ["hello world", 'how are you buddy ?']

tokenizer = text.BertTokenizer(vocab_table, token_out_type=tf.int64, lower_case=do_lower_case)
tokens = tokenizer.tokenize(text_inputs)

# BERT module excepts a 2D tensor (not 3D)
tokens = tokens.to_tensor()[:, :, 0]
tokens = tf.cast(tokens, dtype=tf.int32)

# Set masks and segment ids
input_mask = tf.ones(tokens.shape, dtype=tf.int32)
segment_ids = tf.zeros(tokens.shape, dtype=tf.int32)

# Embed the inputs.
pooled_output, sequence_output = bert_layer([tokens, input_mask, segment_ids])
print(pooled_output)
print(sequence_output)
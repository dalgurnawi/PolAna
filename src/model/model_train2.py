import numpy as np
import tensorflow as tf
import tensorflow_text as tft
from tensorflow import keras
import requests


url = "https://github.com/tensorflow/text/blob/master/tensorflow_text/python/ops/test_data/test_wp_en_vocab.txt?raw=true"
r = requests.get(url)
filepath = "vocab.txt"
open(filepath, 'wb').write(r.content)


tokenizer = tft.BertTokenizer(filepath, token_out_type=tf.string, lower_case=True)
tokens = tokenizer.tokenize(["What you know you can't explain, but you feel it."])
print(tokens.to_list())


raw_inputs = [
    [711, 632, 71],
    [73, 8, 3215, 55, 927],
    [83, 91, 1, 645, 1253, 927],
]

# By default, this will pad using 0s; it is configurable via the
# "value" parameter.
# Note that you could "pre" padding (at the beginning) or
# "post" padding (at the end).
# We recommend using "post" padding when working with RNN layers
# (in order to be able to use the
# CuDNN implementation of the layers).
padded_inputs = tf.keras.preprocessing.sequence.pad_sequences(
    raw_inputs, padding="post"
)
print(padded_inputs)

embedding = keras.layers.Embedding(input_dim=5000, output_dim=16, mask_zero=True)
masked_output = embedding(padded_inputs)

print(masked_output._keras_mask)

masking_layer = keras.layers.Masking()
# Simulate the embedding lookup by expanding the 2D input to 3D,
# with embedding dimension of 10.
unmasked_embedding = tf.cast(
    tf.tile(tf.expand_dims(padded_inputs, axis=-1), [1, 1, 10]), tf.float32
)

masked_embedding = masking_layer(unmasked_embedding)
print(masked_embedding._keras_mask)

model = keras.Sequential(
    [keras.layers.Embedding(input_dim=5000, output_dim=16, mask_zero=True),
     keras.layers.LSTM(32),

     ]
)

model.summary()

# inputs = keras.Input(shape=(None,), dtype="int32")
# x = keras.layers.Embedding(input_dim=5000, output_dim=16, mask_zero=True)(inputs)
# outputs = keras.layers.LSTM(32)(x)
#
# model = keras.Model(inputs, outputs)
#
# model.summary()
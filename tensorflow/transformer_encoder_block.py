"""
This is the transformer endoer block, which will help the model
understand the relationships within inputs. 

This will learn to recognize missing letters in character comparison. 
Defines the Transformer encoder block.
"""

import tensorflow as tf


class TransformerEncoderBlock(tf.keras.layers.Layer):
    def __init__(self, d_model, num_heads, dff):
        super().__init__()

        # look at all positions in the input sequence
        # and learn relationships between characters

        self.self_att = tf.keras.layers.MultiHeadAttention(
            num_heads = num_heads, # parallel heads
            key_dim = d_model)

        # feed forward network -> this part is genuinely not clicking
        # SOS hope to discuss... pulled using ur example and mr. copilot where 
        # i needed it

        self.ffn = tf.keras.Sequential([
            tf.keras.layers.Dense(dff, activation = "relu"),
            tf.keras.layers.Dense(d_model)])

        # stabilizes training

        self.norm1 = tf.keras.layers.LayerNormalization()
        self.norm2 = tf.keras.layers.LayerNormalization()

    def call(self, x):
        attn = self.self_att(x, x)
        x = self.norm1(x + attn)
        ffn_out = self.ffn(x)
        return self.norm2(x + ffn_out)
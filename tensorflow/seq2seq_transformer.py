import tensorflow as tf
from dtsc330.tensorflow.token_position_embedding import TokenPositionEmbedding
from transformer_encoder_block import TransformerEncoderBlock
from transformer_decoder_block import TransformerDecoderBlock


class Seq2SeqTransformer(tf.keras.Model):
    def __init__(self, vocab_size, max_len, d_model=64, num_heads=2, dff=128):
        super().__init__()

        self.embedding = tf.keras.layers.Embedding(vocab_size, d_model)
        self.pos_encoding = TokenPositionEmbedding(max_len + 2, d_model)

        self.encoder = TransformerEncoderBlock(d_model, num_heads, dff)
        self.decoder = TransformerDecoderBlock(d_model, num_heads, dff)

        self.final_layer = tf.keras.layers.Dense(vocab_size)

    def call(self, enc_in, dec_in):
        # encoder
        enc = self.embedding(enc_in)
        enc = self.pos_encoding(enc)
        enc_out = self.encoder(enc)

        # decoder
        dec = self.embedding(dec_in)
        dec = self.pos_encoding(dec)
        dec_out = self.decoder(dec, enc_out)

        return self.final_layer(dec_out)
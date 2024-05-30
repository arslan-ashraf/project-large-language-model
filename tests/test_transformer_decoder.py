import pytest
import tensorflow as tf
from model.transformer_decoder import (
	EmbeddingLayers,
	AttentionHead,
	MaskedMultiHeadAttention,
	FeedForward,
	TransformerDecoderBlock,
	TransformerDecoderModel
)

config = {
	"num_heads": 4,
	"is_masked": True,
	"embedding_dim": 32,
	"hidden_dim": 32,
	"first_layer_size": 4 * 32,
	"droptout_rate": 0.2,
	"vocab_size": 1024,
	"max_seq_length": 5,
	"num_decoder_blocks": 4
}

class TestTransformerDecoderModel:

	def test_EmbeddingLayers(self):
		embed_layers = EmbeddingLayers(config)
		batch_size = 3
		max_seq_length = 5
		input_token_ids = tf.random.uniform(shape=[batch_size, max_seq_length], 
											maxval=1024, 
											dtype=tf.int32)

		embed_output = embed_layers(input_token_ids)

		assert embed_output.shape == (batch_size, max_seq_length, config["hidden_dim"])


	def test_AttentionHead(self):
		input_embeddings = tf.random.uniform(shape=(1, 5, 32))
		head_dim = 8

		attn_head = AttentionHead(head_dim=head_dim, is_masked=True)

		attn_out = attn_head(input_embeddings)

		assert attn_out.shape == (input_embeddings.shape[0], input_embeddings.shape[1], head_dim)


	def test_MaskedMultiHeadAttention(self):
		input_embeddings = tf.random.uniform(shape=(16, 5, 32))

		masked_mha = MaskedMultiHeadAttention(config)
		mha_outputs = masked_mha(input_embeddings)

		assert input_embeddings.shape == mha_outputs.shape


	def test_FeedForward(self):
		input_embeddings = tf.random.uniform(shape=(16, 5, 32))

		feed_forward = FeedForward(config)
		ff_outputs = feed_forward(input_embeddings)
		
		assert input_embeddings.shape == ff_outputs.shape				


	def test_TransformerDecoderBlock(self):
		input_embeddings = tf.random.uniform(shape=(16, 5, 32))

		decoder_block = TransformerDecoderBlock(config)
		decoder_block_outputs = decoder_block(input_embeddings)
		
		assert input_embeddings.shape == decoder_block_outputs.shape


	def test_TransformerDecoderModel(self):
		batch_size = 64
		max_seq_length = 5
		x_train = tf.random.uniform(shape=[batch_size, max_seq_length], 
									maxval=1024, 
									dtype=tf.int32)

		y_train = tf.random.uniform(shape=[batch_size, max_seq_length], 
									maxval=1024, 
									dtype=tf.int32)

		decoder_model = TransformerDecoderModel(config)
		decoder_model_outputs = decoder_model(x_train)
		
		assert decoder_model_outputs.shape == (batch_size, max_seq_length, config["vocab_size"])
		
		loss_fn = tf.keras.losses.SparseCategoricalCrossentropy(from_logits=True)
		decoder_model.compile(loss=loss_fn, optimizer="adam")
		decoder_model.fit(x_train, y_train, epochs=3)
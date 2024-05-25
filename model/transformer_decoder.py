import tensorflow as tf


class AttentionHead(tf.keras.layers.Layer):
	def __init__(self, head_dim, is_masked):
		super().__init__()

		self.query_matrix = tf.keras.layers.Dense(units=head_dim)
		self.key_matrix = tf.keras.layers.Dense(units=head_dim)
		self.value_matrix = tf.keras.layers.Dense(units=head_dim)

		self.is_masked = is_masked

	def call(self, hidden_state):
		queries = self.query_matrix(hidden_state)
		keys = self.key_matrix(hidden_state)
		values = self.value_matrix(hidden_state)

		attention = self.dot_product_attention(queries, keys, values)

		return attention

	def dot_product_attention(self, queries, keys, values):
		key_dim = tf.cast(keys.shape[-1], dtype=tf.float32)
		scores = tf.linalg.matmul(queries, keys, transpose_b=True) / tf.math.sqrt(key_dim)

		if self.is_masked == True:
			seq_length = queries.shape[1]
			ones_matrix = tf.ones(shape=[seq_length, seq_length])
			upper_triangular_matrix = tf.linalg.band_part(ones_matrix, num_lower=0, num_upper=-1)
			ones_diagonal = tf.linalg.band_part(ones_matrix, num_lower=0, num_upper=0)
			_mask = upper_triangular_matrix - ones_diagonal
			mask = _mask * -1e9
			scores += mask

		weights = tf.nn.softmax(scores, axis=-1)

		return tf.linalg.matmul(weights, values)


class MaskedMultiHeadAttention(tf.keras.layers.Layer):
	def __init__(self, config):
		super().__init__()

		self.is_masked = config["is_masked"]
		self.num_heads = config["num_heads"]
		self.embedding_dim = config["embedding_dim"]
		self.head_dim = self.embedding_dim // self.num_heads

		self.attention_heads = [
			AttentionHead(self.head_dim, self.is_masked) for _ in range(self.num_heads)
		]

		self.linear_output = tf.keras.layers.Dense(units=self.embedding_dim)

	def call(self, hidden_state):
		attention_heads_outputs = [
			attention_head(hidden_state) for attention_head in self.attention_heads
		]

		mha_attention = tf.concat(attention_heads_outputs, axis=-1)
		mha_output = self.linear_output(mha_attention)

		return mha_output


class FeedForward(tf.keras.layers.Layer):
	def __init__(self, config):
		super().__init__()

		self.dense1 = tf.keras.layers.Dense(units=config["first_layer_size"], activation="gelu")
		self.dense2 = tf.keras.layers.Dense(units=config["hidden_dim"])
		self.dropout = tf.keras.layers.Dropout(rate=config["droptout_rate"])

	def call(self, x):
		x = self.dense1(x)
		x = self.dense2(x)
		x = self.dropout(x)
		return x


class TransformerDecoderBlock(tf.keras.layers.Layer):
	def __init__(self, config):
		super().__init__()
		self.layer_norm1 = tf.keras.layers.LayerNormalization()
		self.layer_norm2 = tf.keras.layers.LayerNormalization()
		self.mha_attention = MaskedMultiHeadAttention(config)
		self.feed_forward = FeedForward(config)

	def call(self, x):
		# first apply layer norm (to embeddings the first time)
		hidden_state1 = self.layer_norm1(x)

		# skip connection and masked attention
		x = x + self.mha_attention(hidden_state1)
		hidden_state2 = self.layer_norm2(x)

		# skip connection and feed forward
		x = x + self.feed_forward(hidden_state2)

		return x


class EmbeddingLayers(tf.keras.layers.Layer):
	def __init__(self, config):
		super().__init__()

		self.token_embeddings = tf.keras.layers.Embedding(
			input_dim=config["vocab_size"],
			output_dim=config["hidden_dim"]
		)
		self.position_embeddings = tf.keras.layers.Embedding(
			input_dim=config["max_seq_length"],
			output_dim=config["hidden_dim"]
		)
		self.layer_norm = tf.keras.layers.LayerNormalization()
		self.dropout = tf.keras.layers.Dropout(rate=config["droptout_rate"])

	def call(self, input_token_ids):
		
		token_embeddings = self.token_embeddings(input_token_ids)

		seq_length = input_token_ids.shape[1]
		_position_ids = tf.range(seq_length, dtype=tf.float32)
		position_ids = tf.expand_dims(_position_ids, axis=0)
		position_embeddings = self.position_embeddings(position_ids)

		embeddings = token_embeddings + position_embeddings
		embeddings = self.layer_norm(embeddings)
		embeddings = self.dropout(embeddings)

		return embeddings


class TransformerDecoderModel(tf.keras.Model):
	def __init__(self, config):
		super().__init__()
		self.embedding_layers = EmbeddingLayers(config)
		self.decoder_blocks = [
			TransformerDecoderBlock(config) for _ in range(config["num_decoder_blocks"])
		]

	def call(self, input_token_ids):

		"""
		input_token_ids: tensor of shape (batch_size, max_seq_length) contains
		integers mapping to tokens, if max_seq_length exceeds seq_length, then the 
		rest of sequence is padded by zeroes, so inputs always have the same shape
		"""

		x = self.embedding_layers(input_token_ids)
		for decoder_block in self.decoder_blocks:
			x = decoder_block(x)

		return x
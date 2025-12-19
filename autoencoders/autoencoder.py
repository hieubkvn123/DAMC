import tensorflow as tf


class UserAutoEncoder(tf.keras.Model):
    def __init__(self, input_feature_len, latent_dim):
        super(UserAutoEncoder, self).__init__()
        self.latent_dim = latent_dim
        self.input_feature_len = input_feature_len
        
        # Define encoder layers
        self.encoder = tf.keras.Sequential([
            tf.keras.layers.Dense(400, activation='relu'),
            tf.keras.layers.BatchNormalization(),
            # tf.keras.layers.Dropout(0.2),
            tf.keras.layers.Dense(40, activation='relu'),
            tf.keras.layers.BatchNormalization(),
            # tf.keras.layers.Dropout(0.2),
            tf.keras.layers.Dense(self.latent_dim, activation='relu'),
            tf.keras.layers.BatchNormalization(),
            # tf.keras.layers.Dropout(0.2),
        ])

        # Define decoder layers
        self.decoder = tf.keras.Sequential([
            tf.keras.layers.Dense(40, activation='relu'),
            tf.keras.layers.BatchNormalization(),
            # tf.keras.layers.Dropout(0.2),
            tf.keras.layers.Dense(400, activation='relu'),
            tf.keras.layers.BatchNormalization(),
            # tf.keras.layers.Dropout(0.2),
            tf.keras.layers.Dense(self.input_feature_len, activation='softmax')
        ])


    def call(self, inputs):
        encoded = self.encoder(inputs)
        decoded = self.decoder(encoded)
        return decoded

    def encode(self, inputs, training=False):
        return self.encoder(inputs, training=training)

class ItemAutoEncoder(tf.keras.Model):
    def __init__(self, input_feature_len, latent_dim):
        super(ItemAutoEncoder, self).__init__()
        self.latent_dim = latent_dim
        self.input_feature_len = input_feature_len
        
        # Define encoder layers
        self.encoder = tf.keras.Sequential([
            tf.keras.layers.Dense(400, activation='relu'),
            tf.keras.layers.BatchNormalization(),
            # tf.keras.layers.Dropout(0.2),
            tf.keras.layers.Dense(40, activation='relu'),
            tf.keras.layers.BatchNormalization(),
            # tf.keras.layers.Dropout(0.2),
            tf.keras.layers.Dense(self.latent_dim, activation='relu'),
            tf.keras.layers.BatchNormalization(),
            # tf.keras.layers.Dropout(0.2),
        ])

        # Define decoder layers
        self.decoder = tf.keras.Sequential([
            tf.keras.layers.Dense(40, activation='relu'),
            tf.keras.layers.BatchNormalization(),
            # tf.keras.layers.Dropout(0.2),
            tf.keras.layers.Dense(400, activation='relu'),
            tf.keras.layers.BatchNormalization(),
            # tf.keras.layers.Dropout(0.2),
            tf.keras.layers.Dense(self.input_feature_len, activation='softmax')
        ])

    def call(self, inputs):
        encoded = self.encoder(inputs)
        decoded = self.decoder(encoded)
        return decoded

    def encode(self, inputs, training=False):
        return self.encoder(inputs, training=training)








# class BaseAutoEncoder(tf.keras.Model):
#     def __init__(self, input_feature_len, latent_dim, layers_config, regularizer_factor, dropout_rate, use_batch_norm):
#         super(BaseAutoEncoder, self).__init__()
#         self.latent_dim = latent_dim
#         self.input_feature_len = input_feature_len
#         self.layers_config = layers_config
#         self.regularizer_factor = regularizer_factor
#         self.dropout_rate = dropout_rate
#         self.use_batch_norm = use_batch_norm

#         self.build_model()

#     def build_model(self):
#         # Regularizer
#         regularizer = None
#         if self.regularizer_factor is not None:
#             regularizer = tf.keras.regularizers.l2(self.regularizer_factor)

#         # Encoder
#         self.encoder = tf.keras.Sequential()
#         for units in self.layers_config['encoder']:
#             self.encoder.add(tf.keras.layers.Dense(units, activation='relu'))
#             # if self.use_batch_norm:
#             #     self.encoder.add(tf.keras.layers.BatchNormalization())
#             # self.encoder.add(tf.keras.layers.Dropout(self.dropout_rate))

#         # Decoder
#         self.decoder = tf.keras.Sequential()
#         num_decoder_layers = len(self.layers_config['decoder'])

#         for i, units in enumerate(self.layers_config['decoder']):
#             # Determine the activation function: 'relu' for all but the last layer, 'softmax' for the last
#             activation_function = 'relu' if i < num_decoder_layers - 1 else 'softmax'

#             self.decoder.add(tf.keras.layers.Dense(units, activation=activation_function))

# #             # Add batch normalization if enabled
# #             if self.use_batch_norm and activation_function == 'relu':
# #                 self.decoder.add(tf.keras.layers.BatchNormalization())

# #             # Add dropout only if it's not the last layer
# #             if i < num_decoder_layers - 1:
# #                 self.decoder.add(tf.keras.layers.Dropout(self.dropout_rate))

#     def call(self, inputs):
#         encoded = self.encoder(inputs)
#         decoded = self.decoder(encoded)
#         return decoded

#     def encode(self, inputs, training=False):
#         return self.encoder(inputs, training=training)

# class UserAutoEncoder(BaseAutoEncoder):
#     def __init__(self, input_feature_len, latent_dim, encoder_layers, decoder_layers, regularizer_factor=None, dropout_rate=0.2, use_batch_norm=True):
#         layers_config = {
#             'encoder': encoder_layers,
#             'decoder': decoder_layers
#         }
#         super().__init__(input_feature_len, latent_dim, layers_config, regularizer_factor, dropout_rate, use_batch_norm)

# class ItemAutoEncoder(BaseAutoEncoder):
#     def __init__(self, input_feature_len, latent_dim, encoder_layers, decoder_layers, regularizer_factor=None, dropout_rate=0.2, use_batch_norm=True):
#         layers_config = {
#             'encoder': encoder_layers,
#             'decoder': decoder_layers
#         }
#         super().__init__(input_feature_len, latent_dim, layers_config, regularizer_factor, dropout_rate, use_batch_norm)

# class AutoEncoder4Layer(BaseAutoEncoder):
#     def __init__(self, input_feature_len, latent_dim):
#         layers_config = {
#             'encoder': [500, 100, 10],
#             'decoder': [10, 100, 500]
#         }
#         super().__init__(input_feature_len, latent_dim, layers_config)

# class AutoEncoderRobust(BaseAutoEncoder):
#     def __init__(self, input_feature_len, latent_dim, regularizer_factor=0.001):
#         layers_config = {
#             'encoder': [500, 100, 10],
#             'decoder': [10, 100, 500]
#         }
#         super().__init__(input_feature_len, latent_dim, layers_config, regularizer_factor)

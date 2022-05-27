import math

from libs.architecture.target import Autoencoder
import tensorflow as tf


class MEx_CVAEC(tf.keras.Model):
    def __init__(self, layer_dims: list = None):
        """
        "Mixture of experts with convolutional and variational autoencoders for anomaly detection" by Yu et al.
        After: https://link.springer.com/article/10.1007/s10489-020-01944-5
        :param layer_dims: dimensions of the dense encoder
        """
        super(MEx_CVAEC, self).__init__()

        self.layer_dims = layer_dims

        self.exp_1 = None
        self.exp_2 = None
        self.gate = None
        self.tower_1 = None
        self.tower_2 = None

        self.kl_divergence = tf.keras.losses.KLDivergence()
        self.loss = tf.keras.losses.MeanSquaredError()

    def build(self, input_shape):
        # # Instantiate gating network
        self.gate = MEx_CVAEC_AE(layer_dims=self.layer_dims, code_dim_override=4)

        # Instantiate the two experts
        expert_ae = MEx_CVAEC_AE(layer_dims=self.layer_dims)
        expert_ae.build(input_shape=input_shape)
        second_enc = tf.keras.models.clone_model(expert_ae.m_enc)
        second_enc._name = "enc_2"

        enc_outputs = expert_ae.m_enc(expert_ae.m_enc.inputs)
        enc_outputs[0] = tf.keras.layers.Flatten()(enc_outputs[0])
        dec_outputs = expert_ae.m_dec(enc_outputs[-1])
        output = second_enc(dec_outputs[-1])

        expert = tf.keras.Model(inputs=expert_ae.m_enc.inputs, outputs=enc_outputs + dec_outputs + [output[-1]])
        self.exp_1 = expert

        expert_2  = tf.keras.models.clone_model(expert)
        expert_2.build(input_shape=input_shape)

        self.exp_2 = expert_2

        # Instantiate towers
        self.tower_1 = tf.keras.layers.Dense(1, activation='sigmoid')
        self.tower_2 = tf.keras.layers.Dense(1, activation='sigmoid')

    def compile(self, learning_rate=0.00005, loss=tf.keras.losses.MeanSquaredError(), optimizer=None, **kwargs):
        new_optimizer = tf.keras.optimizers.RMSprop(learning_rate)
        return super(MEx_CVAEC, self).compile(optimizer=new_optimizer, loss=loss, **kwargs)

    def train_step(self, train_data):
        x_train = train_data[0]

        with tf.GradientTape(persistent=True) as gradient_tape:
            exp_1_outputs, exp_2_outputs, gate_outputs, tower_1_outputs, tower_2_outputs = self(x_train)
            x_hat_1, z_11, y_hat_1, y_1, z_12 = exp_1_outputs
            x_hat_2, z_21, y_hat_2, y_2, z_22 = exp_2_outputs
            xg_hat, zg, yg_hat, yg = gate_outputs

            batch_size = tf.shape(x_train)[0]
            gaussian = tf.random.normal(shape=(batch_size, z_11.shape[1]), mean=0.5)

            loss_exp_1 = 0.5 * self.compiled_loss(x_train, y_1) + 0.3 * self.compiled_loss(x_hat_1, y_hat_1) + \
                         0.2 * self.kl_divergence(z_11, gaussian)
            loss_exp_z_1 = self.compiled_loss(z_11, z_12)
            loss_1 = 0.7 * loss_exp_1 + 0.3 * loss_exp_z_1

            loss_exp_2 = 0.5 * self.compiled_loss(x_train, y_2) + 0.3 * self.compiled_loss(x_hat_2, y_hat_2) + 0.2 * self.kl_divergence(z_12, gaussian)
            loss_exp_z_2 = self.compiled_loss(z_21, z_22)
            loss_2 = 0.7 * loss_exp_2 + 0.3 * loss_exp_z_2

            loss_gate = 0.7 * self.compiled_loss(x_train, yg) + 0.3 * self.compiled_loss(xg_hat, yg_hat)

            loss_tower = 0.7 * self.compiled_loss(tower_1_outputs, tf.zeros_like(tower_1_outputs)) + 0.3 * self.compiled_loss(tower_2_outputs, tf.zeros_like(tower_2_outputs))

            loss = 0.3 * loss_1 + 0.3 * loss_2 + 0.3 * loss_gate + 0.1 * loss_tower

        recon_grad = gradient_tape.gradient(loss, self.exp_1.trainable_weights + self.exp_2.trainable_weights + self.gate.trainable_weights +
                                            self.tower_1.trainable_weights + self.tower_2.trainable_weights)
        self.optimizer.apply_gradients(zip(recon_grad, self.exp_1.trainable_weights + self.exp_2.trainable_weights + self.gate.trainable_weights+
                                           self.tower_1.trainable_weights + self.tower_2.trainable_weights))
        return {"MEx-CVAEC Loss": loss}

    def call(self, inputs, training=False, mask=None):
        # Call experts
        x_hat_1, z_11, y_hat_1, y_1, z_12 = exp_1_outputs = self.exp_1(inputs)
        x_hat_2, z_21, y_hat_2, y_2, z_22 = exp_2_outputs = self.exp_2(inputs)

        # Call gating
        xg_hat, zg, yg_hat, yg = gate_outputs = self.gate(inputs)

        # Apply sigmoid to middle layer to get scores
        zg = tf.keras.activations.sigmoid(zg)

        # Call towers
        tower_input_1 = tf.expand_dims(zg[:, 0], -1) * z_11 + tf.expand_dims(zg[:, 1], -1) * z_21
        tower_1_outputs = self.tower_1(tower_input_1)

        tower_input_2 = tf.expand_dims(zg[:, 2], -1) * z_12 + tf.expand_dims(zg[:, 3], -1) * z_22
        tower_2_outputs = self.tower_2(tower_input_2)

        return exp_1_outputs, exp_2_outputs, gate_outputs, tower_1_outputs, tower_2_outputs

    def predict_step(self, x_test):
        # x_test = test_data
        exp_1_outputs, exp_2_outputs, gate_outputs, tower_1_outputs, tower_2_outputs = self.call(x_test, training=False)
        x_hat_1, z_11, y_hat_1, y_1, z_12 = exp_1_outputs
        x_hat_2, z_21, y_hat_2, y_2, z_22 = exp_2_outputs

        batch_size = x_test.shape[0]

        S_exp_1 = 0.6 * tf.keras.losses.mean_squared_error(tf.reshape(x_test, shape=(batch_size, -1)), tf.reshape(y_1, shape=(batch_size, -1))) + 0.4 * tf.keras.losses.mean_squared_error(x_hat_1, y_hat_1)
        S_exp_2 = 0.6 * tf.keras.losses.mean_squared_error(tf.reshape(x_test, shape=(batch_size, -1)), tf.reshape(y_2, shape=(batch_size, -1))) + 0.4 * tf.keras.losses.mean_squared_error(x_hat_2, y_hat_2)
        scores = 0.4 * S_exp_1 + 0.4 * S_exp_2 + 0.1 * tf.squeeze(tf.math.pow(tower_1_outputs, 2)) + 0.1 * tf.squeeze(tf.math.pow(tower_2_outputs, 2))

        return tf.reshape(scores, shape=(-1, 1))

    def get_anomaly_score(self, test_data):

        scores = self.predict_step(test_data)

        return scores


class MEx_CVAEC_AE(Autoencoder):

    def __init__(self, **kwargs):
        super(MEx_CVAEC_AE, self).__init__(**kwargs)


    def _conv_encoder(self, input_shape, code_dim=500):
        inputs = tf.keras.layers.Input(shape=input_shape[1:])

        x = tf.keras.layers.Conv2D(filters=32, kernel_size=3)(inputs)
        x = tf.keras.layers.LeakyReLU(0.01)(x)
        x = tf.keras.layers.SpatialDropout2D(self.p_dropout)(x)
        x = tf.keras.layers.BatchNormalization()(x)
        x = tf.keras.layers.MaxPool2D(pool_size=2)(x)

        x = tf.keras.layers.Conv2D(filters=64, kernel_size=3)(x)
        x = tf.keras.layers.LeakyReLU(0.01)(x)
        x = tf.keras.layers.SpatialDropout2D(self.p_dropout)(x)
        x = tf.keras.layers.BatchNormalization()(x)

        x = tf.keras.layers.Conv2D(filters=128, kernel_size=3)(x)
        x = tf.keras.layers.SpatialDropout2D(self.p_dropout)(x)

        x = tf.keras.layers.BatchNormalization()(x)
        x_hat = tf.keras.layers.MaxPool2D(pool_size=2)(x)
        x_hat = tf.keras.layers.Flatten()(x_hat)
        z = tf.keras.layers.Dense(code_dim)(x_hat)

        model = tf.keras.Model(inputs=inputs, outputs=[x_hat, z], name="encoder")
        return model

    def _conv_decoder(self, input_shape, output_dim):
        encoded = tf.keras.Input(shape=input_shape[1][-1])

        y_hat = tf.keras.layers.Dense(2048)(encoded)
        filter = 128
        kernel = int(math.sqrt(input_shape[0][-1] / 128))

        y = tf.keras.layers.Conv2DTranspose(filters=64, kernel_size=2)(tf.reshape(y_hat, shape=[-1, kernel, kernel, filter]))
        y = tf.keras.layers.LeakyReLU(0.01)(y)
        y = tf.keras.layers.SpatialDropout2D(self.p_dropout)(y)
        y = tf.keras.layers.BatchNormalization()(y)
        y = tf.keras.layers.UpSampling2D(size=(2, 2))(y)

        y = tf.keras.layers.Conv2DTranspose(filters=32, kernel_size=3)(y)
        y = tf.keras.layers.LeakyReLU(0.01)(y)
        y = tf.keras.layers.SpatialDropout2D(self.p_dropout)(y)
        y = tf.keras.layers.BatchNormalization()(y)

        y = tf.keras.layers.Conv2DTranspose(filters=1, kernel_size=3)(y)
        y = tf.keras.layers.LeakyReLU(0.01)(y)
        y = tf.keras.layers.SpatialDropout2D(self.p_dropout)(y)
        y = tf.keras.layers.BatchNormalization()(y)
        y = tf.keras.layers.UpSampling2D(size=(2, 2))(y)

        model = tf.keras.Model(inputs=encoded, outputs=[y_hat, y], name="decoded")
        return model

    def _dense_encoder(self, input_shape):
        model = super(MEx_CVAEC_AE, self)._dense_encoder(input_shape=input_shape)

        model = tf.keras.Model(inputs=model.inputs, outputs=[model.layers[-3].output] + model.outputs, name="encoder")
        return model

    def _dense_decoder(self, input_shape, output_dim):
        model = super(MEx_CVAEC_AE, self)._dense_decoder(input_shape=input_shape[1], output_dim=output_dim)

        model = tf.keras.Model(inputs=model.inputs, outputs=[model.layers[0].output] + model.outputs, name="decoder")
        return model

    def build(self, input_shape, code_dim=500):
        self.m_enc = self._conv_encoder(input_shape, code_dim=code_dim if self.code_dim_override is None else self.code_dim_override) if self.layer_dims is None \
            else self._dense_encoder(input_shape)
        self.m_dec = self._conv_decoder(self.m_enc.output_shape, input_shape) if self.layer_dims is None \
            else self._dense_decoder(self.m_enc.output_shape, input_shape)



    def call(self, inputs, training=False, mask=None):
        # Connect the encoder and decoder
        t_encoded = self.m_enc(inputs, training=training, mask=mask)
        t_decoded = self.m_dec(t_encoded[1], training=training, mask=mask)

        return t_encoded + t_decoded


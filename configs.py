class ModelConfig:
    def __init__(self):
        self.device = 'cuda'
        self.conditional = True
        self.num_labels = 10
        self.input_size = 28 * 28
        self.latent_size = 20
        self.dropout = 0.5
        self.initialization = 'normal'
        self.normalization = 'group'
        self.encoder_layers = [1024, 1024]
        self.decoder_layers = [1024, 1024]
        self.activation = 'gelu'


class TrainConfig:
    def __init__(self):
        pass

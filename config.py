import json


class Config():
    """Config class
    """
    def __init__(self, tag):
        self.tag = tag
        self.path = f'runs/{self.tag}'
        self.data_path = ''
        self.mask_coords = []
        self.net_type = 'gan'
        self.l = 128
        self.n_phases = 2
        # Training hyperparams
        self.batch_size = 8
        self.beta1 = 0.9
        self.beta2 = 0.999
        self.num_epochs = 250
        self.iters = 1000
        self.lrg = 0.001
        self.lr = 0.0005
        self.Lambda = 10
        self.critic_iters = 10
        self.pw_coeff = 1e3
        self.lz = 7
        self.lf = 7
        self.dl = 32
        self.ngpu = 2
        if self.ngpu > 0:
            self.device_name = "cuda:0"
        else:
            self.device_name = 'cpu'
        self.nz = 100
        # Architecture
        self.lays = 5
        self.laysd = 5
        # kernel sizes
        self.dk, self.gk = [4]*self.laysd, [4]*self.lays
        self.ds, self.gs = [2]*self.laysd, [2]*self.lays
        self.df, self.gf = [self.n_phases, 64, 128, 256, 512, 1], [
            self.nz, 512, 256, 128, 64, self.n_phases]
        self.dp = [1, 1, 1, 1, 1]
        self.gp = [1, 1, 1, 1, 1]

        # self.gs[0] = 1

    def save(self):
        j = {}
        for k, v in self.__dict__.items():
            j[k] = v
        with open(f'{self.path}/config.json', 'w') as f:
            json.dump(j, f)

    def load(self):
        with open(f'{self.path}/config.json', 'r') as f:
            j = json.load(f)
            for k, v in j.items():
                setattr(self, k, v)

    def get_net_params(self):
        return self.dk, self.ds, self.df, self.dp, self.gk, self.gs, self.gf, self.gp
    
    def get_train_params(self):
        return self.l, self.dl, self.batch_size, self.beta1, self.beta2, self.num_epochs, self.iters, self.lrg, self.lr, self.Lambda, self.critic_iters, self.lz, self.nz



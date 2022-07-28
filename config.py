import json


class Config():
    """Config class
    """
    def __init__(self, tag):
        self.tag = tag
        self.cli = False
        # self.wandb = True
        self.path = f'runs/{self.tag}'
        self.cm = 'gray'
        self.data_path = ''
        self.mask_coords = []
        self.net_type = 'conv-resize'
        self.image_type = 'n-phase'
        self.l = 128
        self.n_phases = 2
        # Training hyperparams
        self.batch_size = 4
        self.beta1 = 0.9
        self.beta2 = 0.999
        self.max_iters = 100e3
        self.timeout = 60*60*12
        self.lrg = 0.0005
        self.lr = 0.0005
        self.Lambda = 10
        self.critic_iters = 10
        self.opt_iters=1e3
        self.finetune_iters=1e4
        self.pw_coeff = 1e2
        self.lz = 7
        self.lf = 7
        self.dl = 32
        self.ngpu = 1
        if self.ngpu > 0:
            self.device_name = "cuda:1"
        else:
            self.device_name = 'cpu'
        self.conv_resize = True
        self.nz = 100
        # Architecture
        self.lays = 5
        self.laysd = 5
        # kernel sizes
        self.dk, self.gk = [4]*self.laysd, [4]*self.lays
        self.ds, self.gs = [2]*self.laysd, [2]*self.lays
        self.df, self.gf = [self.n_phases, 64, 128, 256, 512, 1], [
            self.nz, 512, 256, 128, 64, self.n_phases]
        self.dp, self.gp = [1]*self.laysd, [2]*self.lays
        # Last two layers conv resize (3,1,0)
        self.gk[-2:], self.gs[-2:], self.gp[-2:] = [3, 3], [1,1], [0,0]

    
    def update_params(self):
        self.df[0] = self.n_phases
        self.gf[-1] =  self.n_phases

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
        return self.l, self.dl, self.batch_size, self.beta1, self.beta2, self.lrg, self.lr, self.Lambda, self.critic_iters, self.lz, self.nz


class ConfigPoly(Config):
    def __init__(self, tag):
        super(ConfigPoly, self).__init__(tag)
        self.l = 64
        self.lz = 4
        self.ngpu=1
        self.lays = 5
        self.laysd = 5
        self.frames = 100
        # kernel sizes
        self.dk, self.gk = [4]*self.laysd, [4]*self.lays
        self.ds, self.gs = [2]*self.laysd, [2]*self.lays
        self.df, self.gf = [self.n_phases, 256, 512, 1024, 2048, 1], [
            self.nz, 2048, 1024, 512, 256, self.n_phases]
        self.dp = [1, 1, 1, 1, 0]
        self.gp = [2, 2, 2, 2, 3]
    def get_train_params(self):
        return self.l, self.batch_size, self.beta1, self.beta2, self.lrg, self.lr, self.Lambda, self.critic_iters, self.lz, self.nz
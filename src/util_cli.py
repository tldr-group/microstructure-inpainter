import subprocess
import shutil
import wandb
from dotenv import load_dotenv
import os

class wandbContainer():
    def __init__(self):
        super().__init__()

    def wandb_init(self, name, netG, netD, offline):
        """[summary]

        :param name: [description]
        :type name: [type]
        :param offline: [description]
        :type offline: [type]
        """
        if offline:
            mode = 'disabled'
        else:
            mode = None
        load_dotenv(os.path.join(os.getcwd(), '.env'))
        API_KEY = os.getenv('WANDB_API_KEY')
        ENTITY = os.getenv('WANDB_ENTITY')
        PROJECT = os.getenv('WANDB_PROJECT')
        if API_KEY is None or ENTITY is None or PROJECT is None:
            raise AssertionError('.env file arguments missing. Make sure WANDB_API_KEY, WANDB_ENTITY and WANDB_PROJECT are present.')
        print("Logging into W and B using API key {}".format(API_KEY))
        process = subprocess.run(["wandb", "login", API_KEY], capture_output=True)
        print("stderr:", process.stderr)

        
        print('initing')
        wandb.init(entity=ENTITY, name=name, project=PROJECT, mode=mode)

        wandb_config = {
            'active': True,
            'api_key': API_KEY,
            'entity': ENTITY,
            'project': PROJECT,
            # 'watch_called': False,
            'no_cuda': False,
            # 'seed': 42,
            'log_interval': 1000,

        }
        wandb.watch(netG)
        wandb.watch(netD)
        wandb.config.no_cuda = wandb_config['no_cuda']
        wandb.config.log_interval = wandb_config['log_interval']

    def wandb_save_models(self, fn):
        """[summary]

        :param pth: [description]
        :type pth: [type]
        :param fn: [description]
        :type fn: filename
        """
        shutil.copy(fn, os.path.join(wandb.run.dir, fn))
        wandb.save(fn)
    
    def log(self, args, step):
        wandb.log(args, step=step)
    
    def Image(self, img):
        return wandb.Image(img)
import numpy as np
import imageio
import os
import re 

def sorted_nicely( l ): 
    """ Sort the given iterable in the way that humans expect.""" 
    convert = lambda text: int(text) if text.isdigit() else text 
    alphanum_key = lambda key: [ convert(c) for c in re.split('([0-9]+)', key) ] 
    return sorted(l, key = alphanum_key)

frames = 200
images = []

pth = 'wandb/latest-run/files/media/images'
files = os.listdir(pth)
files = [x for x in files if 'raw' not in x]
files = sorted_nicely(files)[:frames]

logo = imageio.imread('assets/inpainting-logo.png')[::4,::4]
for i in range(60):
    im = imageio.imread(os.path.join(pth,files[0]))
    im[372+128:884-128,672+128:1184-128] = 0
    center = im[372+131:884-131,672+131:1184-131]
    center = np.where(np.repeat(logo[:,:,-1][:,:,np.newaxis],3, axis=2)!=[0,0,0],logo[:,:,:3], [0,0,0])
    im[372+131:884-131,672+131:1184-131] = center
    images.append(im[372:884,672:1184])
for i in range(5):
    images.append(imageio.imread(os.path.join(pth,files[0]))[372:884,672:1184])
for filename in files:
    images.append(imageio.imread(os.path.join(pth,filename))[372:884,672:1184])
    f = filename
for i in range(30):
    images.append(imageio.imread(os.path.join(pth,f))[372:884,672:1184])
imageio.mimsave('assets/movie.gif', images, fps=30)
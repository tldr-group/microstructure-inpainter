import tifffile
import config
import matplotlib.pyplot as plt

c = config.Config('test')
print(c.l)
data = tifffile.imread(c.data_path)
print(data.shape)
mask_factor = 4

mask = data[data.shape[0]//2-c.l//2:data.shape[0]//2+c.l//2, data.shape[0]//2-c.l//2:data.shape[0]//2+c.l//2, data.shape[0]//2-c.l//2:data.shape[0]//2+c.l//2].copy()
unmasked = mask.copy()
mask[c.l//2-c.l//mask_factor:c.l//2+c.l//mask_factor,c.l//2-c.l//mask_factor:c.l//2+c.l//mask_factor,c.l//2-c.l//mask_factor:c.l//2+c.l//mask_factor]=4

print(mask.shape)

fig, axs = plt.subplots(nrows=1, ncols=2)
axs[0].imshow(data[data.shape[0]//2-c.l//2:data.shape[0]//2+c.l//2, data.shape[0]//2-c.l//2:data.shape[0]//2+c.l//2,data.shape[0]//2])
axs[1].imshow(mask[...,c.l//2])
plt.savefig('test.png')


tifffile.imwrite('data/mask.tif', mask)
tifffile.imwrite('data/unmasked.tif', unmasked)
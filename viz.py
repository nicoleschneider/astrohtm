import numpy as np
from astropy.io import fits
from astropy.table import Table

import matplotlib.pyplot as plt

hdu_list = fits.open('test_data.flc')
hdu_list.info()

print(hdu_list[1].columns)
data = Table(hdu_list[1].data)

#for i in data:
 # print(i)
  #break
print(data)
#print(data['ERROR1'])
#NBINS = 100
#hist = plt.hist(data['ERROR1'], NBINS)
#plt.show(hist)

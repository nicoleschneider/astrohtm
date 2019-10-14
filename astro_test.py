from astropy.io import fits
from astropy.io import ascii
from astropy.table import Table


im = fits.open('./test_data.flc')
print(im.info)

data = im[1].data
print(data)


for i, record in enumerate(data[1]):
	print(i, record)

table = [data.field(0), data.field(1)]
ascii.write(table, format='csv')




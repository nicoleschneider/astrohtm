import sys
from astropy.table import Table
import matplotlib.pyplot as plt
import tqdm
import numpy as np


if __name__ == '__main__':
    event_files = sys.argv[1:]
    for event_file in event_files:
        t = Table.read(event_file)
        print(t.colnames)
        print(t.info)

        bin_time = 10
        x_label = 'RAWX'
        y_label = 'RAWY'
		
        t0 = t.meta['TSTART']
        t1 = t.meta['TSTOP']
        new_times = np.arange(t0, t1, bin_time)

        list_of_rows = []
        energy = t['PI']*0.04 + 1.6
        for nt in tqdm.tqdm(new_times):
            good = (t['TIME'] >= nt)&(t['TIME'] < nt + bin_time)
            img, _, _ = np.histogram2d(t[x_label][good], t[y_label][good],
                                       range=((400, 600), (400, 600)), bins=30)
            spec, _ = np.histogram(energy[good], range=[3, 79], bins=30)
            res = {'Time': nt, 'data': img, 'spec': spec}
            list_of_rows.append(res)
        result_table = Table(list_of_rows)
        result_table.write(
            event_file.replace('.gz', '').replace('.evt', '') + '_binned10.fits',
            overwrite=False)

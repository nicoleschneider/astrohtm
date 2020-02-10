# astrohtm

## Project applying anomaly detection algorithms to astronomy data

Algorithm(s):
* Hierarchical Temporal Memory

Data Source(s):
* Light Curve of pulsar

Usage:
* Run bin.py on the .evt file containing the data to aggregate data
  * python bin.py  \<Lower bound of Energy> \<Upper bound of Energy> \<Number of seconds to agregate data by> \<Number of channels to make in the histogram> \<data file>.evt
  
* Run AstroHTM.py on the .fits file produced by bin.py
  * python AstroHTM.py \<Minimum variance per channel cutoff> \<Start time in seconds> \<End time in seconds> \<Name of data file (omit the .fits ending, it is assumed)> \<Filename to store output graph> <Number of channels to expect (same as given to bin.py)>

* (Optional) Run viz.py to customize the visualization of the results
  * python viz.py \<Start time> \<End time> \<Name of csv file where anomaly output was stored (See AstroHTM.py)>

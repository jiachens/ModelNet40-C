# Visualization of ModelNet40-C

## Requirements

* Python3
* Python packages numpy, pandas, matplotlib, seaborn, and open3d (optional).
* Github repository [mitsuba2](https://github.com/mitsuba-renderer/mitsuba2) (optional).

## Steps

* Configuration file `config.py` records the path of ModelNet40-C dataset and the log files of experiments. Please make sure they are refering to right directories.
* Script `main_results.py` processes the log files to collect accuracy results of all experiments. It also draws most of the figures and tables in the paper (Figure 1, 4, 9-16, Table 1-4).
* Script `confusion_matrix.py` draws the confusion matrices in paper (Figure 3, 8).
* Script `examples.py` draws the point cloud examples (Figure 2, 5-7).

All results are saved to `figures` folder by default.

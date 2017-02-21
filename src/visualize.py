import matplotlib.pyplot as plt
import DataScienceBowl as dsb

imgen = dsb.load_sample()

for idx, i in enumerate(imgen):
    dsb.plot_3d(i, 0, idx)

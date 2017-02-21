import matplotlib.pyplot as plt
import DataScienceBowl as dsb

imgen = dsb.load_sample()
for i in imgen:
    dsb.plot_3d(i, 0)

import matplotlib.pyplot as plt
import DataScienceBowl as dsb

imgen = dsb.load_sample(img_size=(200,200,200))

next(imgen)
# for idx, i in enumerate(imgen):
    # dsb.plot_3d(i, 0, idx)

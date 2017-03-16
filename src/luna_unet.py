import SimpleITK as sitk
from glob import glob
import os
import pandas as pd
from tqdm import tqdm
import numpy as np

verify_results = True
if verify_results:
    import matplotlib.pyplot as plt
    plt.ion()
    fig, ax = plt.subplots(2, 2, figsize=[8, 8])

luna_path = "../input/luna_2016"  # TODO fill in path
subset = "subset0"
file_list = glob(os.path.join(luna_path, subset, "*.mhd"))

output_path = subset + "_npy"  # TODO fill in path

# Helper function to associate seriesuid with filename
def get_filename(case):
    global file_list
    for f in file_list:
        if case in f:
            return f


def make_mask(z_px, px_center, mm_center, origin, diam_mm, spacing, width,
              height, padding=5):
    """
    Center : centers of circles mm -- list of coordinates x,y,z
    diam : diameters of circles mm -- diameter
    widthXheight : pixel dim of image
    spacing = mm/px conversion rate np array x,y,z
    origin = x,y,z mm np.array
    """
    slice_arr = np.arange(len(z_px))
    mask = np.zeros((len(z_px), height, width), dtype=np.uint8)  # z, y, x
    rad_mm = diam_mm / 2
    # Get the bounds for the search cube
    # Add 20 to increase the search space
    x_bound, y_bound = np.ceil(rad_mm / spacing[:2] + padding).astype(int)
    # Loop through the cube and set px within diam to 1
    # OLD VERSION
    # for z_px in slice_arr:
    #     for y_px in range(-y_bound + px_center, y_bound + px_center):
    #         for x_px in range(-x_bound + px_center, x_bound + px_center):
    #             pt = np.array([x_px, y_px, z_px])
    #             if np.linalg.norm(spacing * pt + origin - center_mm) <= diam_mm:
    #                 mask[z_px, y_px, x_px] = 1

    y_px = np.arange(-y_bound + px_center[1], y_bound + 1 + px_center[1])
    x_px = np.arange(-x_bound + px_center[0], x_bound + 1 + px_center[0])
    # Create an array of the cartesian product
    pts = np.array(np.meshgrid(z_px, y_px, x_px)).T.reshape(-1, 3).astype(int)

    # Select the points that lie within the sphere
    pts = pts[np.linalg.norm(spacing * pts[:, [2, 1, 0]] +
                            (origin - center_mm), axis=1) <=
                            (rad_mm + np.average(spacing[:2]) * padding)]
    # Assign those points to 1
    mask[np.where(z_px == pts[:, 0:1])[1], pts[:, 1], pts[:, 2]] = 1

    return mask


def verify(imgs, masks, plt):
    for i in range(len(imgs)):
        print("image %d" % i)
        ax[0, 0].imshow(imgs[i], cmap='gray')
        ax[0, 1].imshow(masks[i], cmap='gray')
        ax[1, 0].imshow(imgs[i] * masks[i], cmap='gray')
        input("Hit ENTER to continue...")


# Get the locations of the nodes
df_node = pd.read_csv(os.path.join(luna_path, "CSVFILES", "annotations.csv"))
# Associate the filename with each row
df_node["file"] = df_node["seriesuid"].apply(get_filename)
# Drop all the rows without associated files
df_node = df_node.dropna()

for fcount, img_file in enumerate(tqdm(file_list)):
    print("Getting mask for image file {}".format(os.path.basename(img_file)))
    mini_df = df_node[df_node["file"] == img_file]  # Get all nodules for file
    if len(mini_df) > 0:  # Some files may not have a nodule -- skip these
        # Just use the biggest node TODO Change this to be every node
        biggest_node = np.argmax(mini_df["diameter_mm"].values)
        # Get the coordinates of the center
        node_x = mini_df["coordX"].values[biggest_node]
        node_y = mini_df["coordY"].values[biggest_node]
        node_z = mini_df["coordZ"].values[biggest_node]
        diam_mm = mini_df["diameter_mm"].values[biggest_node]  # diam of nodule (mm)

        # Read in the mhd image file and get nodule and image info
        itk_img = sitk.ReadImage(img_file)
        center_mm = np.array([node_x, node_y, node_z])  # nodule center mm (x,y,z)
        origin = np.array([itk_img.GetOrigin()]).flatten()    # x,y,z Origin in world (mm)
        spacing = np.array(itk_img.GetSpacing())    # spacing of slices in mm/px
        px_center = np.rint((center_mm - origin) / spacing)  # nodule center px

        # Turn the image into a numpy array
        img_array = sitk.GetArrayFromImage(itk_img)  # indexes are z,y,x
        num_slices, height, width = img_array.shape

        # Nodule center is located in px_center[2] slice (z position)
        # We take the 3 slices closest to the center (-1, +0, +1)
        image_inds = np.arange(int(px_center[2]) - 1,
                               int(px_center[2]) + 2).clip(0, num_slices - 1)
        imgs = img_array[image_inds].astype(np.float32)
        masks = make_mask(image_inds, px_center, center_mm, origin, diam_mm,
                           spacing, width, height)

        # Check to make sure our masks and images look right
        if verify_results:
            verify(imgs, masks, plt)

        # Save the numpy arrays
        else:
            np.save(os.path.join(luna_path, output_path, "images_%04d_%04d.npy" % (fcount, biggest_node)), imgs)
            np.save(os.path.join(luna_path, output_path, "masks_%04d_%04d.npy" % (fcount, biggest_node)), imgs)

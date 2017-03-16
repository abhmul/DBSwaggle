import SimpleITK as sitk
from glob import glob
import os
import pandas as pd
from tqdm import tqdm

output_path = "" # TODO fill in path
luna_path = "" # TODO fill in path
subset = "subset0"
file_list = glob(os.path.join(luna_path, subset, "*.mhd"))

# Helper function to associate seriesuid with filename
def get_filename(case):
    global file_list
    for f in file_list:
        if case in f:
            return f


def make_mask(slice_arr, px_center, mm_center, origin, diam_mm, spacing, width, height):
    """
    Center : centers of circles mm -- list of coordinates x,y,z
    diam : diameters of circles mm -- diameter
    widthXheight : pixel dim of image
    spacing = mm/px conversion rate np array x,y,z
    origin = x,y,z mm np.array
    """
    mask = np.zeros((len(slice_arr), height, width), dtype=np.uint8)# z, y, x
    # Get the bounds for the search cube
    x_bound, y_bound = np.trunc(diam_mm / (2 * spacing)).astype(int)
    # Loop through the cube and set px within diam to 1
    # OLD VERSION
    # for z_px in slice_arr:
    #     for y_px in range(-y_bound + px_center, y_bound + px_center):
    #         for x_px in range(-x_bound + px_center, x_bound + px_center):
    #             pt = np.array([x_px, y_px, z_px])
    #             if np.linalg.norm(spacing * pt + origin - center_mm) <= diam_mm:
    #                 mask[z_px, y_px, x_px] = 1
    z_px = slice_arr
    y_px = np.arange(-y_bound + px_center, y_bound + px_center)
    x_px = np.arange(-x_bound + px_center, x_bound + px_center)
    # Create an array of the cartesian product
    pts = np.array(np.meshgrid(z_px, y_px, x_px)).T.reshape(-1, 3)
    # Select the points that lie within the sphere
    pts = pts[np.linalg.norm(spacing * pt + (origin - center_mm), axis=1) <= diam_mm]
    # Assign those points to 1
    mask[pts[:, 0], pts[:, 1], pts[:, 2]] = 1

    return mask



# Get the locations of the nodes
df_node = pd.read_csv(os.path.join(luna_path, "annotations.csv"))
# Associate the filename with each row
df_node["file"] = df_node["seriesuid"].apply(get_filename)
# Drop all the rows without associated files
df_node = df_node.dropna()

for fcount, img_file in enumerate(tqdm(file_list)):
    print("Getting mask for image file {}".format(os.path.basename(img_file)))
    mini_df = df_node[df_node["file"] == img_file] # Get all nodules for file
    if len(mini_df) > 0: # Some files may not have a nodule -- skip these
        # Just use the biggest node TODO Change this to be every node
        biggest_node = np.argmax(mini_df["diameter_mm"].values)
        # Get the coordinates of the center
        node_x = mini_df["coordX"].values[biggest_node]
        node_y = mini_df["coordY"].values[biggest_node]
        node_z = mini_df["coordZ"].values[biggest_node]
        diam_mm = mini_df["diameter_mm"].values[biggest_node] # diam of nodule (mm)

        # Read in the mhd image file and get nodule and image info
        itk_img = sitk.ReadImage(img_file)
        center_mm = np.array([node_x, node_y, node_z]) # nodule center mm (x,y,z)
        origin = np.array([itk_img.GetOrigin()])    # x,y,z Origin in world (mm)
        spacing = np.array(itk_img.GetSpacing())    # spacing of slices in mm/px
        px_center = np.rint((center_mm - origin) / spacing) # nodule center px

        # Turn the image into a numpy array
        img_array = sitk.GetArrayFromImage(itk_img) # indexes are z,y,x
        num_slices, height, width = img_array.shape

        # Nodule center is located in px_center[2] slice (z position)
        # We take the 3 slices closest to the center (-1, +0, +1)
        image_inds = np.arange(int(px_center[2]) - 1,
                               int(px_center[2]) + 2).clip(0, num_slices - 1))
        imgs = img_array[image_inds].astype(np.float32)
        masks  = make_mask(image_inds, px_center, center_mm, origin, diam_mm,
                           spacing, width, height)

        # Save the numpy arrays
        np.save(os.path.join(output_path, "images_%04d_%04d.npy" % (fcount, biggest_node)), imgs)
        np.save(os.path.join(output_path, "masks_%04d_%04d.npy" % (fcount, biggest_node)), imgs))

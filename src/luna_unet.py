import SimpleITK as sitk
from glob import glob
import os
import pandas as pd
from tqdm import tqdm

luna_path = "" # TODO fill in path
subset = "subset0"
file_list = glob(os.path.join(luna_path, subset, "*.mhd"))

# Helper function to associate seriesuid with filename
def get_filename(case):
    global file_list
    for f in file_list:
        if case in f:
            return f

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
        diam = mini_df["diameter_mm"].values[biggest_node]

        # Read in the mhd image file and get nodule and image info
        itk_img = sitk.ReadImage(img_file)
        center = np.array([node_x, node_y, node_z]) # nodule center (x,y,z)
        origin = np.array([itk_img.GetOrigin()])    # x,y,z Origin in world (mm)
        spacing = np.array(itk_img.GetSpacing())    # spacing of slices in mm
        v_center = np.rint((center - origin) / spacing) # nodule center in world

        # Turn the image into a numpy array
        img_array = sitk.GetArrayFromImage(itk_img) # indexes are z,y,x
        num_z, height, width = img_array.shape

        # We're just going to keep 3 slices in z,y,x format
        image_inds = np.arange(int(v_center[2]) - 1,
                               int(v_center[2]) + 2).clip(0, num_z - 1))
        imgs = img_array[image_inds].astype(np.float32)
        masks = np.zeros((3, height, width), dtype=np.uint8)

        # Nodule center is located in v_center[2] slice (z position)
        # We take the 3 slices closest to the center (-1, +0, +1)
        for i, z in enumerate(np.arange(int(v_center[2]) - 1,
                                        int(v_center[2]) + 2).clip(0, num_z - 1)):
            make_mask(center, diam, z * spacing[2] + origin[2],
                      width, height, spacing, origin, inplace=(masks, z))

        # Save the numpy arrays
        np.save(os.path.join(output_path, "images_%04d_%04d.npy" % (fcount, biggest_node)), imgs)
        np.save(os.path.join(output_path, "masks_%04d_%04d.npy" % (fcount, biggest_node)), imgs))

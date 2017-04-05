import numpy as np # linear algebra
import pandas as pd # data processing, CSV file I/O (e.g. pd.read_csv)
import skimage, os
from skimage.morphology import ball, disk, dilation, binary_erosion, remove_small_objects, erosion, closing, reconstruction, binary_closing
from skimage.measure import label,regionprops, perimeter
from skimage.morphology import binary_dilation, binary_opening
from skimage.filters import roberts, sobel
from skimage import measure, feature
from skimage.segmentation import clear_border
from skimage import data
from scipy import ndimage as ndi
import matplotlib.pyplot as plt
from mpl_toolkits.mplot3d.art3d import Poly3DCollection
import dicom
import scipy.misc
import numpy as np
import DataScienceBowl
from subprocess import check_output

def read_ct_scan(folder_name):
        """
        This function takes in a path name to a complete ct scan and outputs the ct scan as an numpy array.
        """
        # Read the slices from the dicom file
        slices = [dicom.read_file(folder_name + filename) for filename in os.listdir(folder_name)]

        # Sort the dicom slices in their respective order
        slices.sort(key=lambda x: int(x.InstanceNumber))

        # Get the pixel values for all the slices
        slices = np.stack([s.pixel_array for s in slices])
        slices[slices == -2000] = 0

        return slices

def plot_ct_scan(scan):
    """
    To visualise the slices, we will have to plot them. matplotlib is used for plotting the slices.
    The plot_ct_scan function takes a 3D CT Scanned Image array as input and plots equally spaced slices.
    The CT Scans are grayscale images i.e. the value of each pixel is a single sample, which means it carries
    only intensity information.
    """
    f, plots = plt.subplots(int(scan.shape[0] / 20) + 1, 4, figsize=(25, 25))
    for i in range(0, scan.shape[0], 5):
        plots[int(i / 20), int((i % 20) / 5)].axis('off')
        plots[int(i / 20), int((i % 20) / 5)].imshow(scan[i], cmap=plt.cm.bone)
    plt.show()

def get_segmented_lungs(input_scan, **kwargs):
    """
    This funtion segments the lungs from the given 2D slice. It will generate
    a binary mask and then apply it to the input 2D scan.
    """
    im = input_scan
    plot = False
    if kwargs is not None and 'plot' in kwargs:
        plot = kwargs['plot']

    if plot == True:
        f, plots = plt.subplots(8, 1, figsize=(5, 40))

    # Step 1: Convert into a binary image.
    binary = im < 604
    if plot == True:
        plots[0].axis('off')
        plots[0].imshow(binary, cmap=plt.cm.bone)

    # Step 2: Remove the blobs connected to the border of the image.
    cleared = clear_border(binary)
    if plot == True:
        plots[1].axis('off')
        plots[1].imshow(cleared, cmap=plt.cm.bone)

    # Step 3: Label the image.
    label_image = label(cleared)
    if plot == True:
        plots[2].axis('off')
        plots[2].imshow(label_image, cmap=plt.cm.bone)

    # Step 4: Keep the labels with 2 largest areas.
    areas = [r.area for r in regionprops(label_image)]
    areas.sort()
    if len(areas) > 2:
        for region in regionprops(label_image):
            if region.area < areas[-2]:
                for coordinates in region.coords:
                       label_image[coordinates[0], coordinates[1]] = 0
    binary = label_image > 0
    if plot == True:
        plots[3].axis('off')
        plots[3].imshow(binary, cmap=plt.cm.bone)

    # Step 5: Erosion operation with a disk of radius 2. This operation is
    # seperate the lung nodules attached to the blood vessels.
    selem = disk(2)
    binary = binary_erosion(binary, selem)
    if plot == True:
        plots[4].axis('off')
        plots[4].imshow(binary, cmap=plt.cm.bone)

    # Step 6: Closure operation with a disk of radius 10. This operation is
    # to keep nodules attached to the lung wall.
    selem = disk(10)
    binary = binary_closing(binary, selem)
    if plot == True:
        plots[5].axis('off')
        plots[5].imshow(binary, cmap=plt.cm.bone)

    # Step 7: Fill in the small holes inside the binary mask of lungs.
    edges = roberts(binary)
    binary = ndi.binary_fill_holes(edges)
    if plot == True:
        plots[6].axis('off')
        plots[6].imshow(binary, cmap=plt.cm.bone)

    # Step 8: Superimpose the binary mask on the input image.
    get_high_vals = binary == 0
    im[get_high_vals] = 0
    if plot == True:
        plots[7].axis('off')
        plots[7].imshow(im, cmap=plt.cm.bone)
        plt.show()

    return im

def segment_lung_from_ct_scan(ct_scan):
    """
    This function will segment an entire lung using the function get_semgented_lung.
    It applies the binary mask to every slice in the lung.
    """
    return np.asarray([get_segmented_lungs(slice) for slice in ct_scan])

def remove_two_largest_cc(segmented_ct_scan):
    """
    This function is to be used after filtering, to remove noise because of blood
    vessels. Removes the two largest connected component.
    Mutates the input segmented ct scan with the largest two ccs removed.
    """

    selem = ball(2)
    binary = binary_closing(segmented_ct_scan, selem)

    label_scan = label(binary)

    areas = [r.area for r in regionprops(label_scan)]
    areas.sort()

    for r in regionprops(label_scan):
        max_x, max_y, max_z = 0, 0, 0
        min_x, min_y, min_z = 1000, 1000, 1000

        for c in r.coords:
            max_z = max(c[0], max_z)
            max_y = max(c[1], max_y)
            max_x = max(c[2], max_x)

            min_z = min(c[0], min_z)
            min_y = min(c[1], min_y)
            min_x = min(c[2], min_x)
        if (min_z == max_z or min_y == max_y or min_x == max_x or r.area > areas[-3]):
            for c in r.coords:
                segmented_ct_scan[c[0], c[1], c[2]] = 0
        else:
            index = (max((max_x - min_x), (max_y - min_y), (max_z - min_z))) / (min((max_x - min_x), (max_y - min_y) , (max_z - min_z)))

def gen_candidates(path_name='../input/sample_images/00cba091fa4ad62cc3200a657aeb957e/', disp=False):
    ct_scan = read_ct_scan(path_name)

    # For testing: display the input lung
    if disp:
        plot_ct_scan(ct_scan)

    # segment entire lung
    segmented_ct_scan = segment_lung_from_ct_scan(ct_scan)

    # Filter the lung based off of known threshold intensity (> 604)
    segmented_ct_scan[segmented_ct_scan < 604] = 0

    # Remove largest ccs
    remove_two_largest_cc(segmented_ct_scan)

    if disp:
        DataScienceBowl.plot_3d(segmented_ct_scan, 604)

    # resize
    # segmented_ct_scan_pixels = DataScienceBowl.get_pixels_hu(segmented_ct_scan)
    segmented_ct_scan, _ = DataScienceBowl.resample(segmented_ct_scan, DataScienceBowl.load_scan(path_name))

    # rescale segmented ct Scan
    segmented_ct_scan = DataScienceBowl.rescale(segmented_ct_scan, 0, 255)
    segmented_ct_scan = segmented_ct_scan.astype(np.uint8)
    return segmented_ct_scan

def gen_segment_all(outpath="segmented_lungs.npy", train_type="sample"):
    """
    This function takes in the path to the ouput file npy form, and the train type and will generate a list
    of segmented lungs for all samples in the given path.

    Returns: None, outputs outpath .npy file.
    """
    data_paths = {
        "train": "../input/train/",
        "sample": "../input/sample/",
        "test": "../input/test/"
    }
    data_path = data_paths[train_type]
    # image_names = [os.path.join(data_path, dirname) for dirname in os.listdir(data_path)]
    # print image_names
    print("Generating all segmented lungs")
    output_lungs = [gen_candidates(path_name=(os.path.join(data_path, dirname)) + "/") for dirname in os.listdir(data_path)]
    np.save(outpath, output_lungs)
    print("Done!")

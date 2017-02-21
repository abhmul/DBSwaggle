import numpy as np # linear algebra
import pandas as pd # data processing, CSV file I/O (e.g. pd.read_csv)
import dicom
import os
import scipy.ndimage

from skimage import measure, morphology


def load_train(new_spacing=[1,1,1], threshold=-320, fill_lung_structures=True, norm=None, center_mean=None):
    train_path = "../input/train/"
    return image_generator(train_path, new_spacing, threshold, fill_lung_structures, norm, center_mean)


def load_sample(new_spacing=[1,1,1], threshold=-320, fill_lung_structures=True, norm=None, center_mean=None):
    sample_path = "../input/sample/"
    return image_generator(sample_path, new_spacing, threshold, fill_lung_structures, norm, center_mean)


def load_test(new_spacing=[1,1,1], threshold=-320, fill_lung_structures=True, norm=None, center_mean=None):
    test_path = "../input/test/"
    return image_generator(test_path, new_spacing, threshold, fill_lung_structures, norm, center_mean)


def image_generator(data_path, new_spacing=[1,1,1], threshold=-320, fill_lung_structures=True, norm=None, center_mean=None):
    """
    Inputs:
        data_path -- Path to directory with images to be loaded/processed.
        new_spacing -- New spacing to assign between pixels in resampling
        fill_lung_structures -- boolean, determines whether or not to fill lungs in segmentation
        norm -- None if no normalization, or else (min_bound, max_bound)
        center_mean -- None if no zero centering, or else pixel_mean

    Returns:
        An image generator that yields the next image in the directory, preprocessing completed.
    """
    # Grab all dicom files from data_path directory
    image_names = [f_name for f_name in os.listdir(data_path) if os.splitext(fname)[1] == ".dcm"]
    for i_name in image_names:
        # Load patient
        slices = load_scan(os.path.join(data_path, i_name))

        # Convert pixels to HU
        pixels = get_pixels_hu(slices)

        # Resample pixels
        resampled_pixels, spacing = resample(pixels, slices, new_spacing)

        # Get segmented lung mask
        segmented_lungs = segment_lung_mask(resampled_pixels, threshold, fill_lung_structures)

        # Normalize and zero center if desired
        if norm is not None and len(norm) == 2:
            segmented_lungs = normalize(segmented_lungs, norm[0], norm[1])
        if center_mean is not None:
            segmented_lungs = zero_center(center_mean
        yield segmented_lungs

# Load the scans in given folder path
def load_scan(path):
    """
    Loads a dicom file specified by path into a list of slices.
    Each slice is a dicom object.
    Infers the slice thickness from the first two slices.
    """
    slices = [dicom.read_file(path + '/' + s) for s in os.listdir(path)]
    slices.sort(key=lambda x: int(x.ImagePositionPatient[2]))
    try:
        slice_thickness = np.abs(slices[0].ImagePositionPatient[2] - slices[1].ImagePositionPatient[2])
    except:
        slice_thickness = np.abs(slices[0].SliceLocation - slices[1].SliceLocation)

    for s in slices:
        s.SliceThickness = slice_thickness

    return slices


def get_pixels_hu(slices):
    """
    Turns each image pixel to Hounsfield units
    Sets pixels outside the bound of the image to 0
    Returns a 3darray of the scan
    """
    image = np.stack([s.pixel_array for s in slices])
    # Convert to int16 (from sometimes int16),
    # should be possible as values should always be low enough (<32k)
    image = image.astype(np.int16)

    # Set outside-of-scan pixels to 0
    # The intercept is usually -1024, so air is approximately 0
    image[image == -2000] = 0

    # Convert to Hounsfield units (HU)
    for slice_number in range(len(slices)):

        intercept = slices[slice_number].RescaleIntercept
        slope = slices[slice_number].RescaleSlope

        if slope != 1:
            image[slice_number] = (slope * image[slice_number].astype(np.float64)).astype(np.int16)

        image[slice_number] += np.int16(intercept)

    return np.array(image, dtype=np.int16)


def resample(image, scan, new_spacing=[1,1,1], mode='nearest'):
    """
    Resizes image so we have uniform spacing between pixels
    Returns a tuple of the image and the new_spacing
    """
    # Determine current pixel spacing
    spacing = np.array([scan[0].SliceThickness] + scan[0].PixelSpacing, dtype=np.float32)

    resize_factor = spacing / new_spacing
    new_real_shape = image.shape * resize_factor
    new_shape = np.round(new_real_shape)
    real_resize_factor = new_shape / image.shape
    new_spacing = spacing / real_resize_factor

    image = scipy.ndimage.interpolation.zoom(image, real_resize_factor, mode='nearest')

    return image, new_spacing

def plot_3d(image, threshold=-300):
    """
    Plots the image in 3d space of all pixels with HU above threshold
    """
    # Position the scan upright,
    # so the head of the patient would be at the top facing the camera
    p = image.transpose(2,1,0)

    verts, faces = measure.marching_cubes(p, threshold)

    fig = plt.figure(figsize=(10, 10))
    ax = fig.add_subplot(111, projection='3d')

    # Fancy indexing: `verts[faces]` to generate a collection of triangles
    mesh = Poly3DCollection(verts[faces], alpha=0.1)
    face_color = [0.5, 0.5, 1]
    mesh.set_facecolor(face_color)
    ax.add_collection3d(mesh)

    ax.set_xlim(0, p.shape[0])
    ax.set_ylim(0, p.shape[1])
    ax.set_zlim(0, p.shape[2])

    plt.show()

def plot_2d(segmented_image, threshold=-300):
    """
    Plots a segmented image in 2d space of all pixels with HU above threshold
    """
    p = segmented_image.transpose(2,1,0)
    for im_slice in p:
        # float32 tells imshow that values are between 0 and 1
        plt.imshow(im_slice.astype(np.float32))
        plt.show()

def largest_label_volume(im, bg=-1):
    """
    im -- a 3d array of the lung scan in HU
    bg -- the background value of the image

    Returns the value other than bg that
    occurs the most and None if there is no such value.
    """
    vals, counts = np.unique(im, return_counts=True)

    counts = counts[vals != bg]
    vals = vals[vals != bg]

    if len(counts) > 0:
        return vals[np.argmax(counts)]
    else:
        return None

def segment_lung_mask(image, threshold=-320, fill_lung_structures=True):

    # not actually binary, but 1 and 2.
    # 0 is treated as background, which we do not want
    binary_image = np.array(image > threshold, dtype=np.int8)+1
    labels = measure.label(binary_image)

    # Pick the pixel in the very corner to determine which label is air.
    #   Improvement: Pick multiple background labels from around the patient
    #   More resistant to "trays" on which the patient lays cutting the air
    #   around the person in half
    background_label = labels[0,0,0]

    #Fill the air around the person
    binary_image[background_label == labels] = 2


    # Method of filling the lung structures (that is superior to something like
    # morphological closing)
    if fill_lung_structures:
        # For every slice we determine the largest solid structure
        for i, axial_slice in enumerate(binary_image):
            axial_slice = axial_slice - 1
            labeling = measure.label(axial_slice)
            l_max = largest_label_volume(labeling, bg=0)

            if l_max is not None: #This slice contains some lung
                binary_image[i][labeling != l_max] = 1


    binary_image -= 1 #Make the image actual binary
    binary_image = 1-binary_image # Invert it, lungs are now 1

    # Remove other air pockets insided body
    labels = measure.label(binary_image, background=0)
    l_max = largest_label_volume(labels, bg=0)
    if l_max is not None: # There are air pockets
        binary_image[labels != l_max] = 0

    return binary_image

def normalize(image, min_bound=-1000, max_bound=400):
    image = (image - min_bound) / (max_bound - min_bound)
    image[image>1] = 1.
    image[image<0] = 0.
    return image

def zero_center(image, pixel_mean = 0.25):
    image = image - pixel_mean
    return image

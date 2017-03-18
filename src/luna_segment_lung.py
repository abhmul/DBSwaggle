import numpy as np
from skimmage import morphology
from skimage import measure
from sklearn.cluster import KMeans
from skimage.transform import resize
from glob import glob
import os

progbar = True
try:
    from tqdm import tqdm
except ImportError:
    tqdm = lambda x: x
    print("Install TQDM for a progress bar")
    progbar = False

luna_path = ""  # TODO fill in path
subset = "subset0_npy"
file_list = glob(os.path.join(luna_path, subset))
test_split = 0.2

print("Creating lungmasks")
for img_file in tqdm(file_list):
    imgs_to_process = np.load(img_file).astype(np.float64)

    if not progbar:
        print("On Image %s" % img_file)

    for i in range(len(imgs_to_process)):
        img = imgs_to_process[i]
        # Standardize the pixel values
        mean = np.mean(img)
        std = np.std(img)
        img = img - mean
        img = img / std
        # Find the average pixel value near the lungs
        # to renormalize washed out images_
        middle = img[100:400, 100:400]  # Arbitrary values
        mean = np.mean(middle)
        img_max = np.max(img)
        img_min = np.min(img)
        # To improve threshold finding, move the underflow
        # and overflow in pixel intensity frequency to the mmiddle mean
        img[img == img_max] = mean
        img[img == img_min] = mean

        # Using KMeans to seperate foregrand (radio-opaque tissue)
        # and background (radio transparent tissue ie lungs)
        # Doing this only on the center of the image to avoid
        # non-tissue parts of the image as much as possible
        kmeans = KMeans(n_cluster=2).fit(middle.reshape((np.prod(middle.shape), 1)))
        centers = np.sort(kmeans.cluster_centers_.flatten())
        # Our threshold falls in between the two kinds of pixels
        threshold = np.mean(centers)
        thresh_img = np.where(img < threshold, 1., 0.) # threshold the image

        # Initial erosion helpful for removing graininess from some of the
        # regions ad then large dilation is used to make the lung regions
        # engulf the vessels and incursions into the lung cavity by
        # radio opaque tissue
        thresh_img = morphology.erosion(thresh_img, np.ones((4,4)))
        thresh_img = morphology.dilation(thresh_img, np.ones((10, 10)))

        # Label each region and obtaion the region properties.
        labels = measure.label(thresh_img)
        label_vals = np.unique(labels)
        regions = measure.regionprops(labels)
        good_labels = []
        for prop in regions:
            B = prop.bbox
            # The background region is removed by removing regions
            # with a bbox that is too large in either dimension.
            # Also, the lungs are generally far away from the top
            # and bottom of the image, so any regions that are too close
            # to the and bottom are removed.
            # This does not produce a perfect segmentation of the lungs from
            # the image, but it is surprisingly good considering its simplicity.
            # Width and height less than 475 and top below 40 and bottom above 472.
            if B[2] - B[0] < 475 and B[3] - B[1] < 475  and B[0] > 40 and B[2] < 472:
                good_labels.append(prop.label)
        mask = sum(labels == N for N in good_labels)
        mask = morphology.dilation(mask, np.ones((10, 10)))  # one last dilation
        imgs_to_process[i] = mask
    np.save(os.path.join(luna_path, subset, img_file.replace("images", "lungmask")), imgs_to_process)

# Now we apply the masks, crop, and resize the image
file_list = glob(os.path.join(luna_path, subset, "lungmask_*.npy"))
out_images = []  # final set of images
out_nodemasks = []  # final set of nodemasks
print("Applying masks to files")
for fname in tqdm(file_list):
    if not progbar:
        print("working on file %s" % fname)

    imgs_to_process = np.load(fname.replace("lungmask", "images"))
    masks = np.load(fname)
    node_masks = np.load(fname.replace("lungmask", "masks"))

    for i in range(len(imgs_to_process)):
        mask = masks[i]
        node_mask = node_masks[i]
        img = imgs_to_process[i]
        img = mask * img  # Aoply the lung mask

        # renormalizing the masked image (in the mask region)
        new_mean = np.mean(img[mask > 0])
        new_std = np.std(img[mask > 0])

        # Pulling the background color up to the lower end
        # of the pixel range for the lungs
        old_min = np.min(img)  # background color
        # After normalization, these pixels will have value -1.2
        img[img == old_min] = new_mean - 1.2 * new_std
        img = img - new_mean
        img = img / new_std
        # make image bounding box (min row, min col, max row, max col)
        labels = measure.label(mask)
        regions = measure.regionprops(labels)
        # Finding the global min and max row over all regions
        # TODO verify this actually works
        min_row = min(512, *(prop.bbox[0] for prop in regions))
        max_row = max(min_row, *(prop.bbox[2] for prop in regions))
        min_col = min(512, *(prop.bbox[1] for prop in regions))
        max_col = max(min_col, *(prop.bbox[3] for prop in regions))
        # Make the image square using the largest dimension
        max_dim = max(max_row - min_row, max_col - min_col)
        if max_dim < 5:
            pass  # Skip all the images with bad regions (5 is arbitrary)
        max_col = min_col + max_dim
        max_row = min_row + max_dim

        # Cropping the image down to the bounding box for all regions
        img = img[min_row:max_row, min_col:max_col]
        mask = mask[min_row:max_row, min_col:max_col]
        node_mask = [min_row:max_row, min_col:max_col]
        if np.all(node_mask == 0):
            pass  # Skip all the images with no node mask
        # Move range to -1 to 1
        img = img + np.min(img)  # 0 to max
        img = 2 * img / np.max(img)  # 0 to 2
        img = img - 1  # -1 to 1

        # resize the image
        new_size = (512, 512)
        new_img = resize(img, new_size)
        new_node_mask = resize(node_mask, new_size)
        # Collect the training images and labels
        out_images.append(new_img)
        out_nodemasks.append(new_node_mask)

# Create the training set and validation set split
num_images = len(out_images)
assert(num_images == len(out_nodemasks))
# Turn them into arrays
out_images = np.stack(out_images, axis=0)
out_nodemasks = np.stack(out_nodemasks, axis=0)

# Shuffle the images
inds = np.random.permutation(num_images)
split_ind = int(len(inds) * (1 - test_split))
np.save(os.path.join(luna_path, "trainImages.npy"), out_images[inds[:split_ind]])
np.save(os.path.join(luna_path, "trainMasks.npy"), out_nodemasks[inds[:split_ind]])
np.save(os.path.join(luna_path, "testImages.npy"), out_images[inds[split_ind:]])
np.save(os.path.join(luna_path, "testMasks.npy"), out_nodemasks[inds[split_ind:]])

import numpy as np
import os
import matplotlib.pyplot as plt
from tqdm import tqdm
from allensdk.core.reference_space_cache import ReferenceSpaceCache
from PIL import Image
import ants
import nrrd
import cv2
from MIFFE import search_path
from scipy.signal import medfilt
from pathlib import Path
from collections import OrderedDict
import tifffile as tiff


def open_AllenSDK(reference_space_key='annotation/ccf_2017', resolution=25):
    # Opens every variable necessary for analysis. 
    # __reference_space_key: key name reference to open a certain annotation (see documentation at allensdk Downloading an annotation volume for other annotations)
    # __resolution: resolution of slices in microns (default is 25)
    # rspc: opens the cache from which tree and annotation will be extracted
    # annotation, meta: downloads the annotation volume to your hard drive if not already done
    # os.listdir is a command that prints the directory in which it is now installed
    # rsp: gets reference space
    # name_map: dictionary of Names to IDs

    rspc = ReferenceSpaceCache(resolution, reference_space_key, manifest='manifest.json')
    tree = rspc.get_structure_tree(structure_graph_id=1) 
    annotation, meta = rspc.get_annotation_volume()

    rsp = rspc.get_reference_space()
    

    return rsp, tree



def map_generator(rsp, tree, show = False, structure = 'all', depth='all'):
    # Creates a vertical projection of superficial IDs in structure ID map (from top to bottom)
    # __rsp: reference space from which the atlas data is taken (like Id maps and names and stuff)
    # __show: if it equals 'yes', it plots the ID_map (WARNING: some IDs are so high they overshadow every other)
    # __structure: if it equals 'all', every region is kept in id_map. Else, structure takes as value a string of the region name to keep and removes every 
    #              subregion not contained in structure. structure can be 'Cerebellum', 'Isocortex', 'Olfactory areas', and much more (see 3D viewer brain map).
    # __depth: 'all' if all brain is scanned. depth = how many millimeter deep in the brain if you want to stop scan at a certain depth in the brain.

    
    y_dim = rsp.annotation.transpose([0,2,1]).shape[0]
    x_dim = rsp.annotation.transpose([0,2,1]).shape[1]
    
    if depth == 'all':
        z_dim = rsp.annotation.transpose([0,2,1]).shape[2]
        
    elif isinstance(depth, (int, float)):
        z_dim = int(7 + (depth*40))
        
    else:
        raise TypeError('Oopsi, depth variable is either "all" or an int corresponding to how deep in the brain you need the scan to be (in mm)')
        

            
    id_map = np.zeros((y_dim,x_dim)) 

    for slice in tqdm(range(z_dim)):
        image = np.squeeze(rsp.annotation.take([slice], axis=1)) #Image is structure id map
        
        np.copyto(id_map, image,where=id_map==0)

    id_list = np.unique(id_map)[1:].astype(int)

    if structure != 'all':

        id_compare = tree.get_structures_by_name([structure])[0].get('id')
        remove_id = []
        keep_id = []
        for i in id_list:
            if not tree.structure_descends_from(i, id_compare):
                remove_id.append(i)
            else:
                keep_id.append(i)

        for i in remove_id:
            id_map = np.where(id_map==i, 0, id_map)

        
    name_map = tree.get_name_map()
    id_name_dict = {}
    for i in keep_id:
        id_name_dict[i] = name_map[i]


    if show:    
        fig, ax = plt.subplots(figsize=(10, 10))
        plt.imshow(id_map, interpolation = 'none',vmax=1300)
        plt.show()

    hardcoded_bregma = (218,228)

    return id_map, id_name_dict, hardcoded_bregma


def create_mask(id_map, i):
    if type(i) == list:
        mask = np.zeros(id_map.shape)
        for j in i:
            mask += np.where(id_map == j, 1, 0)
        return mask
    return np.where(id_map == i, 1, 0).astype(np.uint8)


def create_contour(structure, uint8 = True): #Contours in horizontal (h) and vertical (v) planes. contours are sum of both, and returns boolean (normalized to 1).
    contours_h = abs(np.diff(structure))
    contours_v = abs(np.diff(structure, axis=0))
    contour_h = np.concatenate((contours_h, np.zeros((len(contours_h),1))),axis=1)
    contour_v = np.concatenate((contours_v, np.zeros((1,len(contours_v[0])))))
    contours = contour_h + contour_v
    return np.where(contours!=0, 1, 0).astype(np.uint8)


def identify_files(path, keywords):
    items = os.listdir(path)
    files = []
    for item in items:
        if all(keyword in item for keyword in keywords):
            files.append(item)
    return files


def npy_to_tif(data, name, path = 'search'):
    if path == 'search':
        path = search_path()
    else:
        pass
    data = data.astype('uint8')
    im = Image.fromarray(data)
    im.save(os.path.join(path, name + ".tif"))
    return


def tif_to_nrrd(filename, path):
    img = cv2.imread(path + "/" + filename , cv2.IMREAD_GRAYSCALE)
    filename = Path(filename).stem
    nrrd.write(path + "/" + filename + '.nrrd', img)
    return filename + '.nrrd'


def mask_gradient(mask , n):
    # bottom-up gradient over a mask
    row = np.ones(mask[0].shape)*n
    for i in reversed(range(mask.shape[0])):
        for j in range(mask.shape[1]):
            if mask[i,j] == 1:
                mask[i,j] = row[j]
                if row[j] > 1:
                    row[j] += -1
    return mask


def select_mask(image, colorline = 'black'):
    print(
        'Press "escape" to exit when cropping is done. First and last selected coordinates will automatically connect.')
    # Initialize variables
    roi_points = []
    roi_completed = False

    def mouse_callback(event, x, y, flags, param):
        nonlocal roi_points, roi_completed

        if event == cv2.EVENT_LBUTTONDOWN:
            roi_points.append((x, y))
            if colorline == 'black':
                cv2.circle(image, (x, y), 4, (0, 0, 0), -1)
            if colorline == 'red':
                cv2.circle(image, (x, y), 4, (0, 0, 255), -1)

            if len(roi_points) > 1:
                if colorline == 'black':
                    cv2.line(image, roi_points[-2], roi_points[-1], (0, 0, 0), 5)
                if colorline == 'red':
                    cv2.line(image, roi_points[-2], roi_points[-1], (0, 0, 255), 5)

    # Create a window to display the image
    cv2.namedWindow('Select ROI')
    cv2.imshow('Select ROI', image)

    # Register the mouse callback function
    cv2.setMouseCallback('Select ROI', mouse_callback)
    while not roi_completed:
        cv2.imshow('Select ROI', image)
        key = cv2.waitKey(10)

        if key == 27:
            roi_completed = True

    # Convert the ROI points to a NumPy array
    roi_points = np.array(roi_points)

    # Create a binary mask
    mask = np.zeros_like(image[:, :, 0], dtype=np.uint8)
    cv2.fillPoly(mask, [roi_points], 1)

    cv2.waitKey(10)  # Add a small wait before destroying all windows
    cv2.destroyAllWindows()
    cv2.waitKey(10)  # Add a small wait before destroying all windows
    return mask


import numpy as np
from scipy.signal import medfilt


def intensity_bounds_v2(im, percentile=0.05):
    """
    Adjusts the intensity values of an image based on given percentiles and rescales the image to 8-bit.

    Args:
    im (np.array): The input image loaded with cv2.IMREAD_UNCHANGED.
    percentile (float): The percentile for clipping the intensity values (default is 0.05 for 5-95% range).

    Returns:
    np.array: The rescaled 8-bit image.
    """
    # Calculate the lower and upper percentile bounds
    vmin = np.percentile(im, percentile * 100)
    vmax = np.percentile(im, (1 - percentile) * 100)

    # Clip the image values to the percentile bounds
    clipped_im = np.clip(im, vmin, vmax)

    # Rescale the image to the 0-255 range
    rescaled_im = ((clipped_im - vmin) / (vmax - vmin) * 255).astype(np.uint8)

    return rescaled_im



def obtain_cortical_map(structure='Isocortex'):
    rsp, tree = open_AllenSDK()
    isocortex_map, id_name_dict, _ = map_generator(rsp, tree, structure='Isocortex')
    print('Cortical map obtained.')
    return isocortex_map, id_name_dict


def create_folder(folder_path, new_folder_name):
    new_folder_path = "".join([folder_path, '/', str(new_folder_name)])
    new_folder_path = Path(new_folder_path)

    if not os.path.exists(new_folder_path):
        os.makedirs(new_folder_path)
        print('New folder named "{}" was created.'.format(new_folder_path))
    else:
        print('Folder named "{}" already exists.'.format(new_folder_path))

    return str(new_folder_path)


def overlay(img1, img2):
    # Create a window to display the images
    cv2.namedWindow('Image')

    # Initialize variables
    x_offset = 0
    y_offset = 0
    scale_percent = 100
    dragging = False
    resizing = False
    corner_size = 20
    update_needed = True

    # Define the mouse callback function
    def mouse_callback(event, x, y, flags, param):
        nonlocal x_offset, y_offset, scale_percent, dragging, resizing, update_needed

        if event == cv2.EVENT_LBUTTONDOWN:
            if x > x_offset - corner_size and x < x_offset + corner_size and y > y_offset - corner_size and y < y_offset + corner_size:
                resizing = True
            else:
                dragging = True
        elif event == cv2.EVENT_MOUSEMOVE:
            if dragging or resizing:
                update_needed = True
                if dragging:
                    x_offset += x - param[0]
                    y_offset += y - param[1]
                elif resizing:
                    scale_percent += (x - param[0]) / 10
        elif event == cv2.EVENT_LBUTTONUP:
            dragging = False
            resizing = False

        param[0] = x
        param[1] = y

    # Set the mouse callback function
    cv2.setMouseCallback('Image', mouse_callback, [0, 0])

    while True:
        if update_needed:
            # Resize img2
            width = int(img2.shape[1] * scale_percent / 100)
            height = int(img2.shape[0] * scale_percent / 100)
            dim = (width, height)
            resized_img2 = cv2.resize(img2, dim, interpolation=cv2.INTER_AREA)

            # Create the overlay image
            overlay = img1.copy()
            overlay[y_offset:y_offset + resized_img2.shape[0], x_offset:x_offset + resized_img2.shape[1]] = \
                cv2.addWeighted(
                    overlay[y_offset:y_offset + resized_img2.shape[0], x_offset:x_offset + resized_img2.shape[1]], 0.5,
                    resized_img2, 0.5, 0)

            # Draw the dots at the corners
            cv2.circle(overlay, (x_offset, y_offset), 5, (0, 0, 0), -1)

            # Show the image
            cv2.imshow('Image', overlay)
            update_needed = False

        # Break the loop when the 'q' key is pressed
        if cv2.waitKey(10) & 0xFF == 27:
            break

    # Release the resources and close the window
    cv2.destroyAllWindows()

    return x_offset, y_offset, scale_percent


def crop_and_rescale(transformed_img, reference_img, x_offset, y_offset, scale_percent):
    # rescales images that are transformed and adjusted to masks in 1378 by 1208. That function goes in
    # conjunction with the overlay function. reference_img is the wanted final dimensions. It is necessary
    # to have to descale.

    # calculate the new dimensions of the image
    upsized_width = int(reference_img.shape[1] * scale_percent / 100)
    upsized_height = int(reference_img.shape[0] * scale_percent / 100)
    upsized_dim = (upsized_width, upsized_height)

    # crops leftmost band and uppermost band to have no more offset.
    cropped_img = transformed_img[y_offset: y_offset + upsized_height, x_offset: x_offset + upsized_width]

    # rescales the image
    width = reference_img.shape[1]
    height = reference_img.shape[0]

    dim = (width, height)

    resized_img = cv2.resize(cropped_img, dim, interpolation=cv2.INTER_LINEAR)
    resized_img[resized_img > 0] = 1

    return resized_img


def ants_transformation(ants_surgery_mask, ants_isocortex_mask):
    # Perform Rigid Registration
    rigid = ants.registration(fixed=ants_surgery_mask, moving=ants_isocortex_mask,
                              type_of_transform='Rigid',
                              reg_iterations=(200, 200, 200, 10),
                              aff_iterations=(200, 200, 200, 10),
                              aff_shrink_factors=(12, 8, 4, 2),
                              aff_smoothing_sigmas=(4, 3, 2, 1),
                              metric='MI')

    # Perform Affine Registration
    affine = ants.registration(fixed=ants_surgery_mask, moving=ants_isocortex_mask,
                               initial_transform=rigid['fwdtransforms'][0],
                               type_of_transform='Affine',
                               reg_iterations=(200, 200, 200, 10),
                               aff_shrink_factors=(8, 4, 2, 1),
                               aff_smoothing_sigmas=(4, 3, 2, 1),
                               metric='MI')

    # Perform SyN Registration
    syn = ants.registration(fixed=ants_surgery_mask, moving=ants_isocortex_mask,
                            initial_transform=affine['fwdtransforms'][0],
                            type_of_transform='SyNOnly',
                            reg_iterations=(200, 200, 200, 200, 100, 10, 5),
                            metric='CC')

    return syn


def change_nrrd_headers(reg_files_path, res_fact):
    data1, header1 = nrrd.read(reg_files_path + '/surgery_mask.nrrd')

    header1 = OrderedDict([('type', 'uint8'),
                           ('dimension', 2),
                           ('space dimension', 2),
                           ('sizes', np.array([1378, 1208])),
                           ('space directions',
                            np.array([[round(25. / res_fact, 2), 0.],
                                      [0., round(25. / res_fact, 2)]])),
                           ('encoding', 'raw'),
                           ('space units', ['microns', 'microns'])])

    nrrd.write(reg_files_path + '/surgery_mask.nrrd', data1, header1)

    data2, header2 = nrrd.read(reg_files_path + '/isocortex_mask.nrrd')

    header2 = OrderedDict([('type', 'uint8'),
                           ('dimension', 2),
                           ('space dimension', 2),
                           ('sizes', np.array([456, 528])),
                           ('space directions',
                            np.array([[25., 0.],
                                      [0., 25.]])),
                           ('encoding', 'raw'),
                           ('space units', ['microns', 'microns'])])

    nrrd.write(reg_files_path + '/isocortex_mask.nrrd', data2, header2)


def mask_folder(data_folder, ants_surgery_mask, adjusted_brain_mask, window_image, overlay_parameters, isocortex_map,
                mask_list, transform, overwrite=False):
    # Creates a folder in which movie is transformed to
    mask_folder_name = 'Masks'
    new_mask_path = "".join([data_folder, '/', mask_folder_name])
    new_mask_path = Path(new_mask_path)

    if not os.path.exists(new_mask_path):
        os.makedirs(new_mask_path)

    new_mask_list = []
    masks = []

    for i in mask_list:
        if i != 0:
            file_path = new_mask_path / f'mask_{int(i)}.npy'
            if file_path.exists() and not overwrite:
                print(f"File {file_path} already exists and overwrite is set to False. Skipping.")
                continue

            mask = create_mask(isocortex_map, i)
            mask[mask != 0] = 1
            ants_mask = ants.from_numpy(mask, spacing=(25., 25.))
            transformed_mask = ants.apply_transforms(fixed=ants_surgery_mask,
                                                     moving=ants_mask,
                                                     transformlist=transform['fwdtransforms'])

            transformed_mask = transformed_mask * adjusted_brain_mask

            if len(np.unique(transformed_mask.numpy())) > 1:

                new_mask_list.append(int(i))
                numpy_transformed_mask = transformed_mask.numpy()
                numpy_transformed_mask[numpy_transformed_mask != 0] = 1

                # resize transformed_mask on right scale (the window scale):
                x_offset, y_offset, scale_percent = overlay_parameters
                resized_transformed_mask = crop_and_rescale(numpy_transformed_mask, window_image, x_offset, y_offset,
                                                            scale_percent)

                # Adds the transformed mask to a list of masks that are not yet adjusted to not overlap.
                masks.append(resized_transformed_mask)
            else:
                continue

    # Obtains the overlap_map of all different transformed regions and segments them to assure no
    # overlapping.

    no_overlap_map = np.zeros((window_image.shape[0], window_image.shape[1]))

    for i in range(len(masks)):
        mask = masks[i]
        no_overlap_map += mask

    no_overlap_map[no_overlap_map > 1] = 0

    for i, mask_id in enumerate(new_mask_list):
        mask = masks[i] * no_overlap_map
        tiff.imwrite(str(new_mask_path) + '/mask_{}.tif'.format(int(mask_id)), mask)

    print('Done!')
    return None


def resolution_factor(norm_mask, comp_mask):
    # gives resolution_factor between two different dimension masks

    def widest_row(mask):
        widest = 0
        n = 0
        for i in range(mask.shape[0]):
            nonzero_count = np.count_nonzero(mask[i, :])
            if np.count_nonzero(mask[i, :]) > n:
                widest = i
                n = nonzero_count
        return widest, n

    _, norm = widest_row(norm_mask)
    _, comp = widest_row(comp_mask)
    return norm / comp


def intensity_bounds(im, kernelsize=11, percentile=0.01):
    # Removes pixel intensity aberations, finds optimized bounds for a certain image. kernelsize is for filter
    # and percentile is to determine what's the intensity threshold of a normalized intensity distribution.
    flat = im.flatten()
    flat = medfilt(flat, kernel_size=kernelsize)
    min_range, max_range = np.min(flat), np.max(flat)
    flat_mean, flat_std = np.mean(flat), np.std(flat)

    def gaussian(x, mean, std):
        return np.exp(-(x - mean) ** 2 / (2 * std ** 2))

    x = np.linspace(min_range, max_range, 1000)
    y = gaussian(x, flat_mean, flat_std)

    value_range = np.where(y > percentile)

    vmin = value_range[0][0]
    vmax = x[value_range[0][-1]]

    filtered_im = np.copy(im)
    filtered_im[filtered_im > vmax] = vmax

    rescaled_im = (filtered_im / vmax * 255).astype(np.uint8)

    return rescaled_im, vmin, vmax


def registration(surgery_img_path, data_folder, data_file, overwrite = False, display_first = False):
    '''
    REGISTRATION_PATH is the folder in which every surgery image will be put under the folder associated with a
    specific mouse. The folder is named after the mouse ID like so : registration_folder_path / M38 / Surgery.png .
    DATA_FOLDER is the path to the data experiment folder.
    DATA_FILE is the path to the data experiment file (.tif format).

    '''

    # outputs a cortical map of the allen institute 2017 annotation. Every id in the map are in mask_list.
    isocortex_map, id_name_dict = obtain_cortical_map()
    mask_list = list(id_name_dict.keys())

    # outputs the mask of the allen isocortex
    isocortex_mask = create_mask(isocortex_map, mask_list)

    # obtain the surgery mask

    surgery_image = cv2.imread(surgery_img_path)

    # SELECTS A MASK and outputs a binary npy

    surgery_mask = select_mask(np.copy(surgery_image))

    # Creates a folder named Registration_files inside of data_folder. Registration_files will regroup every
    # item used for the registration process.

    reg_files_path = create_folder(data_folder, 'Registration_files')

    # SAVES the mask in .tif
    surgery_mask_name = 'surgery_mask'
    npy_to_tif(surgery_mask, surgery_mask_name, path=reg_files_path)

    # SAVES isocortex_mask in .tif
    isocortex_mask_name = 'isocortex_mask'
    npy_to_tif(isocortex_mask, isocortex_mask_name, path=reg_files_path)

    # SAVES everything as .nrrd
    im3 = cv2.imread(reg_files_path + '/isocortex_mask.tif', cv2.IMREAD_GRAYSCALE)
    nrrd.write(reg_files_path + "/isocortex_mask.nrrd", im3)

    im4 = cv2.imread(reg_files_path + '/surgery_mask.tif', cv2.IMREAD_GRAYSCALE)
    nrrd.write(reg_files_path + "/surgery_mask.nrrd", im4)

    # 1st registration step: registers isocortex_mask vs surgery_mask with ANTSpy
    res_fact = resolution_factor(surgery_mask, isocortex_mask)

    change_nrrd_headers(reg_files_path, res_fact)
    ants_surgery_mask = ants.image_read(reg_files_path + "/surgery_mask.nrrd")
    ants_isocortex_mask = ants.image_read(reg_files_path + "/isocortex_mask.nrrd")
    transform = ants_transformation(ants_surgery_mask, ants_isocortex_mask)

    if display_first:
        # Apply Transforms
        warped_isocortex_mask = ants.apply_transforms(fixed=ants_surgery_mask,
                                                      moving=ants_isocortex_mask,
                                                      transformlist=transform['fwdtransforms'])

        fig, ax = plt.subplots(figsize=(8, 8), dpi=80)
        plt.imshow(warped_isocortex_mask.numpy(), alpha=0.5, cmap='Reds')
        plt.imshow(ants_surgery_mask.numpy(), alpha=0.5, cmap='Blues')
        plt.show()

    # 2nd registration step: Overlays surgery image and data on one another ton link them by a transform.

    window_layer = cv2.imread(data_file)
    window_layer = intensity_bounds_v2(window_layer)

    # overlay_parameters is a tuple of x_offset, y_offset and scale_percent parameters for the mask_folder function.
    overlay_parameters = overlay_v2(surgery_image, window_layer)

    # Creates a brain mask that will crop masks to only include regions inside the brain window.

    brain_mask = select_mask(np.copy(window_layer), colorline='red')

    x_offset, y_offset, scale_percent = overlay_parameters
    width = int(window_layer.shape[1] * scale_percent / 100)
    height = int(window_layer.shape[0] * scale_percent / 100)
    dim = (width, height)

    resized = cv2.resize(brain_mask, dim, interpolation=cv2.INTER_LINEAR)

    adjusted_brain_mask = np.zeros((surgery_mask.shape[0], surgery_mask.shape[1]))
    adjusted_brain_mask[y_offset:y_offset + resized.shape[0], x_offset:x_offset + resized.shape[1]] = resized[:, :]

    mask_folder(data_folder, ants_surgery_mask, adjusted_brain_mask, window_layer, overlay_parameters, isocortex_map,
                mask_list, transform, overwrite=overwrite)

def identify_files(path, keywords):
    items = os.listdir(path)
    files = []
    for item in items:
        if all(keyword in item for keyword in keywords):
            files.append(item)
    return files


def normalize_image(img):
    """
    Normalize the image to the 0-255 range based on percentiles.

    Args:
    img (np.array): The input image.
    percentile (float): The percentile for clipping the intensity values.

    Returns:
    np.array: The normalized image.
    """
    vmin = np.min(img)
    vmax = np.max(img)

    # Rescale the image to the 0-255 range
    normalized_img = ((img - vmin) / (vmax - vmin) * 255).astype(np.uint8)

    return normalized_img


def overlay_v2(img1, img2):
    # Normalize both images to 0-255 range
    img1_normalized = normalize_image(img1)
    img2_normalized = img2



    # Create a window to display the images
    cv2.namedWindow('Image')

    # Initialize variables
    x_offset = 0
    y_offset = 0
    scale_percent = 100
    dragging = False
    resizing = False
    corner_size = 20
    update_needed = True
    vmin=0
    vmax=255

    # Define the mouse callback function
    def mouse_callback(event, x, y, flags, param):
        nonlocal x_offset, y_offset, scale_percent, dragging, resizing, update_needed

        if event == cv2.EVENT_LBUTTONDOWN:
            if x > x_offset - corner_size and x < x_offset + corner_size and y > y_offset - corner_size and y < y_offset + corner_size:
                resizing = True
            else:
                dragging = True
        elif event == cv2.EVENT_MOUSEMOVE:
            if dragging or resizing:
                update_needed = True
                if dragging:
                    x_offset += x - param[0]
                    y_offset += y - param[1]
                elif resizing:
                    scale_percent += (x - param[0]) / 10
        elif event == cv2.EVENT_LBUTTONUP:
            dragging = False
            resizing = False

        param[0] = x
        param[1] = y

    # Set the mouse callback function
    cv2.setMouseCallback('Image', mouse_callback, [0, 0])

    while True:
        if update_needed:
            # Resize img2
            width = int(img2_normalized.shape[1] * scale_percent / 100)
            height = int(img2_normalized.shape[0] * scale_percent / 100)
            dim = (width, height)
            resized_img2 = cv2.resize(img2_normalized, dim, interpolation=cv2.INTER_AREA)

            # Create the overlay image
            overlay = img1_normalized.copy()
            y1, y2 = y_offset, y_offset + resized_img2.shape[0]
            x1, x2 = x_offset, x_offset + resized_img2.shape[1]

            if y1 < 0 or x1 < 0 or y2 > overlay.shape[0] or x2 > overlay.shape[1]:
                continue

            overlay[y1:y2, x1:x2] = cv2.addWeighted(overlay[y1:y2, x1:x2], 0.5, resized_img2, 0.5, 0)

            # Draw the dots at the corners
            cv2.circle(overlay, (x_offset, y_offset), 5, (0, 0, 0), -1)

            # Show the image
            cv2.imshow('Image', overlay)
            update_needed = False

        # Break the loop when the 'q' key is pressed
        if cv2.waitKey(10) & 0xFF == 27:
            break

    # Release the resources and close the window
    cv2.destroyAllWindows()

    return x_offset, y_offset, scale_percent
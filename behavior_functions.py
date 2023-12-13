import freenect
import cv2
import numpy as np

import time
import matplotlib.pyplot as plt
import matplotlib.animation as animation
from collections import deque
import os
import sys
import pickle

from tqdm.notebook import tqdm
from numba import njit, prange
import tifffile as tiff
from sklearn.linear_model import LinearRegression





# ---------------------------------------------------------------------------------------------------------------------
# ---------------------------------------------------------------------------------------------------------------------
# ---------------------------------------------------------------------------------------------------------------------
# ---------------------------------------------------------------------------------------------------------------------
# ---------------------------------------------------------------------------------------------------------------------


# General and recording functions


def record_video(folder_path, duration, overwrite=False):
    if not overwrite:
        if os.path.exists(folder_path + '/' + 'video.avi'):
            raise OSError('Oops, .avi video file already exists and overwrite arg set to False.')
        if os.path.exists(folder_path + '/' + 'depth.avi'):
            raise OSError('Oops, .avi depth file already exists and overwrite arg set to False.')

    activate_video, _ = freenect.sync_get_video()
    activate_depth, _ = freenect.sync_get_depth()

    del activate_video, activate_depth

    fourcc = cv2.VideoWriter_fourcc(*'MJPG')
    video_out = cv2.VideoWriter(folder_path + '/' + 'video.avi', fourcc, 20.0, (640, 480))
    depth_out = cv2.VideoWriter(folder_path + '/' + 'depth.avi', fourcc, 20.0, (640, 480))

    print('loop starts')

    start_time = time.time()

    while time.time() - start_time < duration:
        frame_video, _ = freenect.sync_get_video()
        frame_video = cv2.cvtColor(frame_video, cv2.COLOR_RGB2BGR)

        frame_depth, _ = freenect.sync_get_depth()
        frame_depth_8bit = cv2.convertScaleAbs(frame_depth, alpha=(255.0 / 2047.0))
        frame_depth_3channel = cv2.merge((frame_depth_8bit, frame_depth_8bit, frame_depth_8bit))

        video_out.write(frame_video)
        depth_out.write(frame_depth_3channel)

        horizontal_image = np.hstack((frame_depth_3channel, frame_video))

        # Simple Downsample
        cv2.imshow('Depth & Video', horizontal_image)
        cv2.waitKey(5)

    video_out.release()
    depth_out.release()

    cv2.waitKey(10)
    cv2.destroyAllWindows()
    cv2.waitKey(10)

    print('Done!')

    return None

def avi_to_array(video_path, greyscale=True):
    # Capture the video from the given path
    cap = cv2.VideoCapture(video_path)

    # Check if video opened successfully
    if not cap.isOpened():
        print("Error: Could not open video.")
        return None

    # Read the first frame to get the dimensions
    ret, frame = cap.read()

    # Check if a frame has been successfully grabbed
    if not ret:
        print("Error: Could not read frame.")
        cap.release()
        return None

    # Convert frame to grayscale if needed
    if greyscale:
        frame = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)

    # Initialize a 3D or 4D array depending on the greyscale parameter
    if greyscale:
        video_array = np.empty((int(cap.get(cv2.CAP_PROP_FRAME_COUNT)), frame.shape[0], frame.shape[1]),
                               dtype=frame.dtype)
    else:
        video_array = np.empty((int(cap.get(cv2.CAP_PROP_FRAME_COUNT)), frame.shape[0], frame.shape[1], frame.shape[2]),
                               dtype=frame.dtype)

    # Set the first frame
    video_array[0] = frame

    # Read the rest of the frames
    for i in range(1, video_array.shape[0]):
        ret, frame = cap.read()
        if not ret:
            print("Error: Could not read frame at index", i)
            break

        # Convert frame to grayscale if needed
        if greyscale:
            frame = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)

        # Append frame to the video array
        video_array[i] = frame

    # Release the video capture object
    cap.release()

    return video_array


def view_avi(filename):
    cap = cv2.VideoCapture(filename)

    while cap.isOpened():
        ret, frame = cap.read()

        if not ret:
            break

        cv2.imshow('Video', frame)

        # Break the loop on pressing 'q'
        if cv2.waitKey(25) & 0xFF == ord('q'):
            break

    cap.release()
    cv2.waitKey(10)
    cv2.destroyAllWindows()
    cv2.waitKey(10)

    return None


def identify_files(path, keywords):
    items = os.listdir(path)
    files = []
    for item in items:
        if all(keyword in item for keyword in keywords):
            files.append(item)
    return files


def dataset_paths(behavior_path, session_keywords, mouse_keywords, verbose=True):
    # Transforms "all" into a list of the right form containing every possibility of SX and MXX in behavior datasets
    # that start with "BH".
    if session_keywords == "all":
        session_list = identify_files(behavior_path, "BH")

        session_keywords = []
        for i in range(len(session_list)):
            session_keywords.append(session_list[i].split('_')[1])

        uniqu_array = np.unique(session_keywords)  # for some reason np.unique orders str with numbers in them.
        session_keywords = list(uniqu_array)


    if mouse_keywords == "all":
        mouse_keywords = []
        for session in session_keywords:
            session_path = behavior_path + "/BH_{}".format(session)
            mouse_names = identify_files(session_path, session)
            for mouse_name in mouse_names:
                mouse_keywords.append(mouse_name.split('_')[1])
        unique_array = np.unique(mouse_keywords)
        mouse_keywords = list(unique_array)

    if verbose:
        print('Mouse keywords: ', mouse_keywords)
        print('Session keywords: ', session_keywords)

    dataset_list = []
    for session in session_keywords:
        for mouse in mouse_keywords:
            data_path = behavior_path + "/BH_{s}/BH_{m}_{s}".format(s=session, m=mouse)
            if os.path.exists(data_path):
                dataset_list.append(data_path)

    return dataset_list


# ---------------------------------------------------------------------------------------------------------------------
# ---------------------------------------------------------------------------------------------------------------------
# ---------------------------------------------------------------------------------------------------------------------
# ---------------------------------------------------------------------------------------------------------------------
# ---------------------------------------------------------------------------------------------------------------------


# Video processing functions


def find_background_mode(data_path, video_type="video", verbose=True, save=True, skip=1):
    if video_type =="video":
        video_path = os.path.join(data_path, "cropped_video.avi")
    elif video_type =="depth":
        video_path = os.path.join(data_path, "cropped_depth.avi")
    else:
        raise ValueError("video_type must be 'video' or 'depth'.")

    if verbose:
        print('Loading video...')
    video = cv2.VideoCapture(video_path)
    N_frames = int(video.get(cv2.CAP_PROP_FRAME_COUNT))
    video.set(1, 0)
    _, frame = video.read()
    if verbose:
        print('Finding background...')
    counts = np.zeros((frame.shape[0], frame.shape[1], 256)) # Only works for 8-bit videos
    video.set(1, 0)
    if skip > 1:
        N_selected = int(N_frames / skip)
        selected_frames = np.linspace(0, N_frames, N_selected, endpoint=False).astype('int')
        for i in tqdm(selected_frames, file=sys.stdout):
            video.set(1, i)
            _, frame = video.read()
            counts = update_counts(counts, frame[:, :, 0].astype(np.uint8))
    else:
        for _ in tqdm(range(1, N_frames), file=sys.stdout):
            _, frame = video.read()
            counts = update_counts(counts, frame[:, :, 0])
    video.release()
    background = np.argmax(counts, axis=2).astype(np.uint8)
    if verbose:
        print('Done! \n')

    if save:
        if video_type == "video":
            background_path = os.path.join(data_path, "background.tif")
        elif video_type == "depth":
            background_path = os.path.join(data_path, "depth_background.tif")

        cv2.imwrite(background_path, background)
        if verbose:
            print(f"Background saved as {background_path}")

    return None

@njit
def update_counts(counts, frame):
    for i in range(counts.shape[0]):
        for j in range(counts.shape[1]):
            counts[i, j, frame[i, j]] += 1
    return counts


def substract_background_with_array(data_path):
    video_path = data_path + "/cropped_video.avi"
    background_path = data_path + "/background.tif"

    cap = cv2.VideoCapture(video_path)
    if not cap.isOpened():
        print("Error: Could not open input video file.")
        return

    background = tiff.imread(background_path)

    fps = int(cap.get(cv2.CAP_PROP_FPS))
    width = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
    height = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))

    fourcc = cv2.VideoWriter_fourcc(*'MJPG')
    video_name = os.path.split(video_path)[1]
    output_video_path = os.path.join(os.path.dirname(video_path), "bg_sub_" + video_name)
    out = cv2.VideoWriter(output_video_path, fourcc, fps, (width, height))

    while True:
        ret, frame = cap.read()
        if not ret:
            break

        frame_gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
        background_substracted_frame = cv2.absdiff(frame_gray, background)
        blurred_bsf = cv2.GaussianBlur(background_substracted_frame, (3, 3), 1)

        out.write(blurred_bsf)

    cap.release()
    out.release()
    print(f"Output video saved to {output_video_path}")




@njit(parallel=True)
def temporal_mean_filter(stack):
    _, height, width = stack.shape
    mean_filter = np.zeros((height, width))
    for i in prange(height):
        for j in prange(width):
            mean_filter[i, j] = np.mean(stack[:, i, j])
    return mean_filter


@njit(parallel=True)
def temporal_max_filter(stack):
    _, height, width = stack.shape
    max_filter = np.zeros((height, width))
    for i in prange(height):
        for j in prange(width):
            max_val = stack[0, i, j]
            for k in prange(stack.shape[0]):
                if stack[k, i, j] > max_val:
                    max_val = stack[k, i, j]
            max_filter[i, j] = max_val
    return max_filter



def centroid_mean_filter(centroids, window_size):
    # Handle the case where window_size is even by incrementing it
    window_size = window_size + 1 if window_size % 2 == 0 else window_size
    half_window = window_size // 2

    # Pad the array with replicated boundary values along the time axis
    padded_array = np.pad(centroids, pad_width=((half_window, half_window), (0, 0)), mode='edge')

    # Initialize the filtered array
    filtered_centroids = np.zeros_like(centroids, dtype=float)

    # Apply the mean filter to each coordinate separately
    for coord in range(centroids.shape[1]):
        for i in range(centroids.shape[0]):
            filtered_centroids[i, coord] = np.mean(padded_array[i:i + window_size, coord])

    return filtered_centroids


def centroid_plot(centroids):
    x, y = centroids[0]
    t = np.linspace(0, centroids.shape[0], centroids.shape[0])

    # Create two subplots (axes) one over the other
    fig, axs = plt.subplots(2, 1, figsize=(8, 6))

    # Plot data on the first subplot
    axs[0].plot(t, centroids[:, 0], color='b')
    axs[0].set_title('x coords')

    # Plot data on the second subplot
    axs[1].plot(t, centroids[:, 1], color='r')
    axs[1].set_title('y coords')

    # Adjust layout and show the plots
    plt.tight_layout()
    plt.show()

def animate_trajectory(video_path, coords, save=False):
    # Set up figure for animation
    fig, ax = plt.subplots(figsize=(8, 8))

    # Open video_array
    video_array = avi_to_array(video_path)

    # mirror y axis to make it work with cv2 coordinate system
    coordinates = np.copy(coords)
    coordinates[:, 1] = -coords[:, 1] + video_array.shape[1]

    # Set up the line and point for the animated plot
    line, = ax.plot([], [], '-', lw=1.5, alpha=0.5, c='r')
    pt, = ax.plot([], [], 'o', c="r", ms=3)

    # Initialize history
    history_x = deque(maxlen=video_array.shape[0])
    history_y = deque(maxlen=video_array.shape[0])

    # Set axes limits
    ax.set_xlim(0, video_array.shape[2])
    ax.set_ylim(0, video_array.shape[1])

    # Define update function for animation
    def update(i):
        print("Frame Index:", i)

        # Get the current frame from the video_array
        frame = video_array[i]

        # Display video frame with the correct extent
        ax.imshow(frame, extent=[0, video_array.shape[2], 0, video_array.shape[1]], cmap='gray')

        if i == 0:
            history_x.clear()
            history_y.clear()

        x, y = coordinates[:, i]
        print("X, Y Coords:", x, y)

        history_x.appendleft(x)
        history_y.appendleft(y)

        # Update the data for the trajectory line and point
        line.set_data(history_x, history_y)
        pt.set_data(x, y)

        fig.canvas.draw()
        return line, pt

    # Create the animation object
    anim = animation.FuncAnimation(fig, update, frames=video_array.shape[0], interval=100, repeat=False)

    if save:
        # Save the animation
        filename = "animation.mp4"
        anim.save(filename, writer='ffmpeg', fps=30)
    else:
        # Display the animation
        plt.tight_layout()
        plt.show()

    return None


def crop_data(arr, coords, radius_of_crop, mask_value=0):
    T, M, N = arr.shape
    cy, cx = coords[1], coords[0]
    R = radius_of_crop
    y, x = np.ogrid[-cy:M - cy, -cx:N - cx]

    mask = x * x + y * y <= R * R
    mask_3d = mask[np.newaxis, :, :]  # Adds an extra dimension
    mask_3d = np.broadcast_to(mask_3d, (T, M, N))  # Broadcasts to shape (T, M, N)
    arr[~mask_3d] = mask_value

    not_masked_rows = np.any(arr != mask_value, axis=(0, 2))
    not_masked_cols = np.any(arr != mask_value, axis=(0, 1))

    first_row = np.where(not_masked_rows)[0][0]
    last_row = np.where(not_masked_rows)[0][-1]
    first_col = np.where(not_masked_cols)[0][0]
    last_col = np.where(not_masked_cols)[0][-1]
    roi = [first_row, last_row + 1, first_col, last_col + 1]

    return arr[:, roi[0]:roi[1], roi[2]:roi[3]]


def crop_video_and_depth(behavior_folder, coords=[], radius_of_crop=130):
    video_path = os.path.join(behavior_folder, "video.avi")
    depth_path = os.path.join(behavior_folder, "depth.avi")

    arr = avi_to_array(video_path)
    first_frame = arr[0, :, :]

    # Select and crop the first frame
    if len(coords) == 0:
        coords = select_center(first_frame)

    elif not isinstance(coords, np.ndarray):
        raise ValueError("Coords must be handed individually as np.arrays.")

    arr = crop_data(arr, coords, radius_of_crop=radius_of_crop)

    # Define the codec and create VideoWriter object
    fourcc = cv2.VideoWriter_fourcc(*'MJPG')
    video_name = os.path.split(video_path)[1]
    output_video_path = os.path.join(os.path.dirname(video_path), "cropped_" + video_name)
    out = cv2.VideoWriter(output_video_path, fourcc, 31.0, (arr.shape[2], arr.shape[1]))

    # Process and write the frames with tqdm
    for i in tqdm(range(arr.shape[0]), desc="Processing video"):
        frame = arr[i]
        rgb_frame = np.repeat(frame[:, :, np.newaxis], 3, axis=2)
        out.write(rgb_frame)

    # Release the video capture and writer objects
    out.release()

    del arr

    print(f"Output video saved to {output_video_path}")

    depth_arr = avi_to_array(depth_path)

    # Apply the transformation to the entire stack
    transformed_depth_arr = []
    affine_matrix = np.array([[0.92, 0.], [0., 0.92]])
    offset = np.array([13., 34.])
    #     affine_matrix = np.array([[1.10195914, 0.        ], [0.        , 1.10195914]])
    #     offset = np.array([-41.77996883, -19.86813022])
    affine_matrix_cv2 = np.hstack([affine_matrix, offset.reshape(-1, 1)])

    for frame in tqdm(depth_arr, desc="Transforming depth to video space"):
        transformed_frame = cv2.warpAffine(frame, affine_matrix_cv2, (frame.shape[1], frame.shape[0]))
        transformed_depth_arr.append(transformed_frame)
    transformed_depth_arr = np.array(transformed_depth_arr)

    cropped_depth_arr = crop_data(transformed_depth_arr, coords, radius_of_crop=radius_of_crop)

    # Initialize VideoWriter for depth video
    # Make sure output_depth_path is defined
    fourcc2 = cv2.VideoWriter_fourcc(*'MJPG')
    output_depth_path = os.path.join(behavior_folder, "cropped_depth.avi")
    out2 = cv2.VideoWriter(output_depth_path, fourcc2, 31.0, (cropped_depth_arr.shape[2], cropped_depth_arr.shape[1]))

    # Process and write the depth frames with tqdm
    for i in tqdm(range(cropped_depth_arr.shape[0]), desc="Processing depth"):
        frame = cropped_depth_arr[i]
        rgb_frame = np.repeat(frame[:, :, np.newaxis], 3, axis=2)
        out2.write(rgb_frame)

    # Release the video writer objects
    out2.release()
    print(f"Output depth saved to {output_depth_path}")

    return None


def select_center(video_path, verbose=True):
    import cv2  # Make sure to import cv2

    # Open the input video file
    cap = cv2.VideoCapture(video_path)
    if not cap.isOpened():
        print("Error: Could not open input video file.")
        return

    # Read the first frame
    ret, first_frame = cap.read()
    if not ret:
        print("Error: Could not read the first frame.")
        cap.release()
        return

    # Initialize variables for storing coordinates
    coords = {"x": None, "y": None}
    click_detected = False  # Flag for click detection

    # Callback function to detect mouse click
    def mouse_click(event, x, y, flags, param):
        nonlocal click_detected
        if event == cv2.EVENT_LBUTTONDOWN:
            coords["x"], coords["y"] = x, y
            click_detected = True  # Set the flag to True when clicked
            cv2.destroyAllWindows()
            if verbose:
                print("Mouse clicked at:", x, y)

    # Create a window to display the image
    cv2.namedWindow('First Frame')
    cv2.setMouseCallback('First Frame', mouse_click)

    # Display the first frame and wait for the click
    while not click_detected:  # Loop until click is detected
        cv2.imshow('First Frame', first_frame)
        if cv2.waitKey(1) & 0xFF == ord('q'):  # Break loop if 'q' is pressed
            break

    cv2.waitKey(1)
    cv2.destroyAllWindows()
    cap.release()

    coords_array = np.array([coords["x"], coords["y"]])
    # Return the coordinates
    return coords_array


def binarize_video(data_path):
    file_path = os.path.join(data_path, "bg_sub_cropped_video.avi")
    output_path = os.path.join(data_path, "mask_video.tif")

    cap = cv2.VideoCapture(file_path)
    if not cap.isOpened():
        print(f"Error: Could not open file {file_path}.")
        return

    def get_first_frame_and_count(file_path):
        cap = cv2.VideoCapture(file_path)
        if not cap.isOpened():
            print(f"Error: Could not open file {file_path}.")
            return None, None
        ret, first_frame = cap.read()
        if not ret:
            print("Error: Could not read the first frame.")
            cap.release()
            return None, None
        total_frames = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
        cap.release()
        return first_frame, total_frames

    def collect_pixel_values(file_path, skip_frames=1):
        cap2 = cv2.VideoCapture(file_path)
        pixel_values = []

        total_frames = int(cap2.get(cv2.CAP_PROP_FRAME_COUNT))

        for i in tqdm(range(0, total_frames, skip_frames)):
            cap2.set(cv2.CAP_PROP_POS_FRAMES, i)
            ret, frame = cap2.read()
            if not ret:
                break

            gray_frame = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
            pixel_values.extend(gray_frame.flatten())

        cap2.release()
        return pixel_values

    pixel_values = collect_pixel_values(file_path, skip_frames=30)
    threshold_value = int(np.percentile(pixel_values, 98.6))

    first_frame, total_frames = get_first_frame_and_count(file_path)

    kernel = cv2.getStructuringElement(cv2.MORPH_CROSS, (3, 3))

    with tiff.TiffWriter(output_path) as tif:
        for i in tqdm(range(total_frames), desc=f"Binarizing {os.path.basename(output_path)}"):
            ret, frame = cap.read()
            if not ret:
                break

            # Convert to grayscale
            gray_frame = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)

            # Apply threshold
            _, binary_mask = cv2.threshold(gray_frame, threshold_value, 255, cv2.THRESH_BINARY)

            # Erode 2 times
            eroded_mask = cv2.erode(binary_mask, kernel, iterations=4)

            # Dilate 5 times
            dilated_mask = cv2.dilate(eroded_mask, kernel, iterations=5)

            # Erode 3 times
            final_mask = cv2.erode(dilated_mask, kernel, iterations=1)

            # Convert to binary (0s and 1s)
            final_mask = (final_mask // 255).astype(np.uint8)

            tif.write(final_mask)

    cap.release()
    print(f"Output saved to {output_path}")


def compute_centroids(dataset_path):
    mask_stack_path = dataset_path + "/" + "mask_video.tif"
    with tiff.TiffFile(mask_stack_path) as tif:
        # Get the number of frames in the stack
        n_frames = len(tif.pages)

        # Initialize an array to store the centroid coordinates
        centroids = np.zeros((n_frames, 2), dtype=np.int16)

        # Process the stack frame by frame
        for i in tqdm(range(n_frames)):
            # Read the current frame
            frame = tif.pages[i].asarray()

            # Find the contours in the binary mask
            contours, _ = cv2.findContours(frame, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)

            # Compute the centroid for the largest contour
            if contours:
                largest_contour = max(contours, key=cv2.contourArea)
                M = cv2.moments(largest_contour)
                if M["m00"] != 0:
                    cx = int(M['m10'] / M['m00'])
                    cy = int(M['m01'] / M['m00'])
                    centroids[i, :] = [cx, cy]

    return centroids


def save_mask_depth(dataset_path):
    # Paths
    mask_path = dataset_path + "/mask_video.tif"
    depth_path = dataset_path + "/cropped_depth.avi"

    # Opening data
    depth_cap = cv2.VideoCapture(depth_path)
    if not depth_cap.isOpened():
        print("Error: Could not open depth video file.")
        return

    fps = int(depth_cap.get(cv2.CAP_PROP_FPS))
    width = int(depth_cap.get(cv2.CAP_PROP_FRAME_WIDTH))
    height = int(depth_cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
    total_frames = int(depth_cap.get(cv2.CAP_PROP_FRAME_COUNT))

    # Prepare video writer to save the output
    out = cv2.VideoWriter(dataset_path + '/depth_mask.avi', cv2.VideoWriter_fourcc(*'MJPG'), fps, (width, height))

    # Read the mask video using tifffile
    with tiff.TiffFile(mask_path) as tif:
        # Get the number of frames in the mask video
        n_frames = len(tif.pages)

        # Check if total_frames matches n_frames
        if total_frames != n_frames:
            print("Error: mask and depth videos don't have matching frame number.")
            return

        # Process each frame
        for i in tqdm(range(total_frames)):
            mask_frame = tif.pages[i].asarray()
            ret, depth_frame = depth_cap.read()
            depth_frame = depth_frame[:, :, 0]

            if not ret:
                print("Error: Could not read frame from depth video.")
                break

            # Combine mask and depth frames
            combined_frame = cv2.multiply(mask_frame, depth_frame)
            combined_frame = cv2.cvtColor(combined_frame, cv2.COLOR_GRAY2BGR)

            # Write the combined frame to the output video file
            out.write(combined_frame)

    # Release the video writer and capture objects
    out.release()
    depth_cap.release()
    print("Processing complete. Result saved as 'depth_mask.avi' in dataset path.")
    return None


def obtain_depth(dataset_path, edge=10):


    def measure_floor_depth(dataset_path):

        # Paths and Measures depth background mode.
        cropped_depth_path = dataset_path + "/" + "cropped_depth.avi"
        depth_background_path = dataset_path + "/" + "depth_background.tif"
        if not os.path.exists(depth_background_path):
            find_background_mode(dataset_path, video_type="depth", verbose=False)

        # Measures floor
        background = tiff.imread(depth_background_path).astype(np.float32)
        background[background > 120] = np.nan
        height, width = background.shape[0], background.shape[1]
        high_ground = np.ceil(np.nanmean(background[height // 3:2 * (height // 3), width // 3:2 * (width // 3)]))
        low_ground = np.floor(np.nanmean(background[height // 3:2 * (height // 3), width // 3:2 * (width // 3)]))

        return low_ground, high_ground

    depth_mask_path = dataset_path + "/depth_mask.avi"
    # Opening data
    cap = cv2.VideoCapture(depth_mask_path)
    if not cap.isOpened():
        print("Error: Could not open depth video file.")
        return

    fps = int(cap.get(cv2.CAP_PROP_FPS))
    width = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
    height = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
    total_frames = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))

    _, high_floor = measure_floor_depth(dataset_path)

    lower_bound = high_floor - edge
    upper_bound = high_floor + 1

    # Process each frame
    mean_values = np.zeros(total_frames)
    last_value = None
    for i in tqdm(range(total_frames)):
        ret, frame = cap.read()
        frame = frame[:, :, 0]

        # Create a mask for values within the specified range
        range_mask = (frame >= lower_bound) & (frame <= upper_bound)

        # Calculate the mean of values within the range
        if np.any(range_mask):
            mean_value = frame[range_mask].mean()
            last_value = mean_value
        else:
            mean_value = last_value

        mean_values[i] = mean_value

    return mean_values


def count_pixels(dataset_path):
    depth_mask_path = dataset_path + "/" + "depth_mask.avi"
    cap = cv2.VideoCapture(depth_mask_path)
    if not cap.isOpened():
        print("Error: Could not open depth video file.")
        return

    total_frames = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
    a = np.zeros(total_frames)
    for i in tqdm(range(total_frames)):
        ret, frame = cap.read()
        gray_scale_frame = frame[:, :, 0]
        a[i] = np.count_nonzero(gray_scale_frame)
    return a


def detrend_linear2d(x_coords, y_coords, z_coords, verbose=False):
    z = np.copy(z_coords)

    # Ajustement du modèle à vos données en x
    model = LinearRegression()
    model.fit(x_coords.reshape(-1, 1), z_coords)
    r_squared_x = model.score(x_coords.reshape(-1, 1), z_coords)

    # Ajustement du modèle à vos données en y
    model2 = LinearRegression()
    model2.fit(y_coords.reshape(-1, 1), z_coords)
    r_squared_y = model2.score(y_coords.reshape(-1, 1), z_coords)
    if verbose:
        print("x ", r_squared_x)
        print("y ", r_squared_y)

    if r_squared_x < r_squared_y:

        fit_z_x = model.predict(x_coords.reshape(-1, 1))
        regression = fit_z_x - np.min(fit_z_x)
        z -= regression
        fit_z_y = model2.predict(y_coords.reshape(-1, 1))
        regression = fit_z_y - np.min(fit_z_y)
        z -= regression

    elif r_squared_x >= r_squared_y:
        fit_z_y = model2.predict(y_coords.reshape(-1, 1))
        regression = fit_z_y - np.min(fit_z_y)
        z -= regression
        fit_z_x = model.predict(x_coords.reshape(-1, 1))
        regression = fit_z_x - np.min(fit_z_x)
        z -= regression

    return z


def save_dict(file_path, data_dict):
    """
    Serialize and save a dictionary to a file using pickle.

    Args:
        file_path (str): The file path where the dictionary will be saved.
        data_dict (dict): The dictionary to be saved.
    """
    with open(file_path, 'wb') as file:
        pickle.dump(data_dict, file)


def load_dict(file_path):
    """
    Load a dictionary from a file saved using pickle.

    Args:
        file_path (str): The file path of the saved dictionary.

    Returns:
        dict: The loaded dictionary.
    """
    with open(file_path, 'rb') as file:
        loaded_dict = pickle.load(file)
    return loaded_dict
# ---------------------------------------------------------------------------------------------------------------------
# ---------------------------------------------------------------------------------------------------------------------
# ---------------------------------------------------------------------------------------------------------------------
# ---------------------------------------------------------------------------------------------------------------------
# ---------------------------------------------------------------------------------------------------------------------

# ARCHIVED

def archived_crop_video_and_depth(behavior_folder, rectangle=None):
    def get_first_frame_and_count(file_path):
        cap = cv2.VideoCapture(file_path)
        if not cap.isOpened():
            print(f"Error: Could not open file {file_path}.")
            return None, None
        ret, first_frame = cap.read()
        if not ret:
            print("Error: Could not read the first frame.")
            cap.release()
            return None, None
        total_frames = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
        cap.release()
        return first_frame, total_frames

    def select_rectangle(image):
        cv2.namedWindow('Select ROI')
        cv2.imshow('Select ROI', image)
        rectangle = cv2.selectROI('Select ROI', image)
        cv2.waitKey(10)
        cv2.destroyAllWindows()
        cv2.waitKey(10)
        return rectangle

    def crop_image(image, rectangle):
        return image[int(rectangle[1]):int(rectangle[1] + rectangle[3]),
               int(rectangle[0]):int(rectangle[0] + rectangle[2])]

    def process_video(file_path, output_path, rectangle):
        cap = cv2.VideoCapture(file_path)
        if not cap.isOpened():
            print(f"Error: Could not open file {file_path}.")
            return
        total_frames = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
        fourcc = cv2.VideoWriter_fourcc(*'MJPG')
        out = cv2.VideoWriter(output_path, fourcc, 31.0, (rectangle[2], rectangle[3]))
        for _ in tqdm(range(total_frames), desc=f"Processing {os.path.basename(output_path)}"):
            ret, frame = cap.read()
            if not ret:
                break
            cropped_frame = crop_image(frame, rectangle)
            out.write(cropped_frame)
        cap.release()
        out.release()
        print(f"Output saved to {output_path}")

    def process_depth(file_path, output_path, rectangle):
        affine_matrix = np.array([[0.92, 0.], [0., 0.92]])
        offset = np.array([13., 34.])
        affine_matrix_cv2 = np.hstack([affine_matrix, offset.reshape(-1, 1)])

        cap = cv2.VideoCapture(file_path)
        if not cap.isOpened():
            print(f"Error: Could not open file {file_path}.")
            return
        total_frames = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
        fourcc = cv2.VideoWriter_fourcc(*'MJPG')
        out = cv2.VideoWriter(output_path, fourcc, 30.0, (rectangle[2], rectangle[3]))
        for _ in tqdm(range(total_frames), desc=f"Processing {os.path.basename(output_path)}"):
            ret, frame = cap.read()
            if not ret:
                break
            frame = cv2.warpAffine(frame, affine_matrix_cv2, (frame.shape[1], frame.shape[0]))
            cropped_frame = crop_image(frame, rectangle)
            out.write(cropped_frame)
        cap.release()
        out.release()
        print(f"Output saved to {output_path}")

    video_path = os.path.join(behavior_folder, "bg_sub_video.avi")
    depth_path = os.path.join(behavior_folder, "depth.avi")

    first_frame, total_frames_video = get_first_frame_and_count(video_path)
    _, total_frames_depth = get_first_frame_and_count(depth_path)
    if first_frame is None:
        return

    if rectangle is None:
        roi = select_rectangle(first_frame)
        print(roi)
    else:
        roi = rectangle

    output_video_path = os.path.join(os.path.dirname(video_path), "cropped_video.avi")
    output_depth_path = os.path.join(os.path.dirname(depth_path), "cropped_depth.avi")

    process_video(video_path, output_video_path, roi)
    process_depth(depth_path, output_depth_path, roi)


def archived_substract_background(video_path, background_path):
    video_path += "/video.avi"
    background_path += "/video.avi"

    bg_stack = avi_to_array(background_path)
    background = np.mean(bg_stack, axis=0).astype(np.uint8)

    cap = cv2.VideoCapture(video_path)
    if not cap.isOpened():
        print("Error: Could not open input video file.")
        return

    fps = int(cap.get(cv2.CAP_PROP_FPS))
    width = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
    height = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
    total_frames = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))

    fourcc = cv2.VideoWriter_fourcc(*'MJPG')
    video_name = os.path.split(video_path)[1]
    output_video_path = os.path.join(os.path.dirname(video_path), "bg_sub_" + video_name)
    out = cv2.VideoWriter(output_video_path, fourcc, fps, (width, height), isColor=False)

    while True:
        ret, frame = cap.read()
        if not ret:
            break

        frame_gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
        background_substracted_frame = cv2.absdiff(frame_gray, background)
        blurred_bsf = cv2.GaussianBlur(background_substracted_frame, (3, 3), 1)

        out.write(blurred_bsf)

    cap.release()
    out.release()
    print(f"Output video saved to {output_video_path}")


def archived_video_substract_background(video_path, temp_filt="mean"):
    # Open the input video file
    cap = cv2.VideoCapture(video_path)
    if not cap.isOpened():
        print("Error: Could not open input video file.")
        exit()

    # Get the video properties
    fps = int(cap.get(cv2.CAP_PROP_FPS))
    width = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
    height = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))

    def spacial_median_filter(img, median_k):
        # Apply the median filter. median_k is the actual height and width of the median filter. not the radius
        filtered_image = cv2.medianBlur(img, median_k)
        return filtered_image

    def gaussian_blur(img, gauss_k, sigma=1):
        # Apply a gaussian filter. gauss_k is the actual height and width of the median filter. not the radius
        return cv2.GaussianBlur(img, (gauss_k, gauss_k), sigma)

    array = avi_to_array(video_path)

    if temp_filt == "max":  # used when experiment is short (1-2 mins)
        background_with_artefacts = temporal_max_filter(array)  # outside function because of @njit usage.
        background = spacial_median_filter(background_with_artefacts, 5)  # 5 is tested on Fiji but might need modif.

    elif temp_filt == "mean":  # used when experiment is long (10-20 mins)
        background = np.mean(array, axis=0)

    else:
        raise ValueError("temp_filt: Only 'max' and 'mean' temporal filters implemented.")

    del array

    # Define the codec and create VideoWriter object
    fourcc = cv2.VideoWriter_fourcc(*'MJPG')
    video_name = os.path.split(video_path)[1]
    output_video_path = os.path.join(os.path.dirname(video_path), "background_substracted_" + video_name)
    out = cv2.VideoWriter(output_video_path, fourcc, fps, (width, height))

    # Process the video frame by frame
    while True:
        ret, frame = cap.read()

        if not ret:
            break

        # Remove 4th dimension (colors)
        frame = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)

        # Apply your filters/transformations to the frame here
        background_substracted_frame = np.abs(frame - background)
        blurred_bsf = gaussian_blur(background_substracted_frame, 3)
        uint8_blurred_bsf = np.uint8(blurred_bsf)
        uint8_blurred_bsf = cv2.cvtColor(uint8_blurred_bsf, cv2.COLOR_GRAY2BGR)

        # Write the modified frame to the output video
        out.write(uint8_blurred_bsf)

    # Release the video capture and writer objects

    cap.release()
    out.release()
    print(f"Output video saved to {output_video_path}")

    return None


def archived_rescale_video(target_video_path, reference_video_path):
    """
    Rescale the target video to match the dimensions of the reference video.

    Parameters:
    - target_video_path: the path to the video you want to resize.
    - reference_video_path: the path to the reference video.
    - output_video_path: the path to save the resized video.
    """

    def get_video_ref_details(video_path):

        cap = cv2.VideoCapture(video_path)
        fps = int(cap.get(cv2.CAP_PROP_FPS))
        width = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
        height = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
        num_frames = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
        cap.release()
        return fps, width, height, num_frames

    # Get dimensions of the reference video
    ref_fps, ref_width, ref_height, num_frames = get_video_ref_details(reference_video_path)

    # Open target video
    cap = cv2.VideoCapture(target_video_path)

    # Video Writer setup
    fourcc = cv2.VideoWriter_fourcc(*'MJPG')
    video_name = os.path.split(target_video_path)[1]
    output_video_path = os.path.join(os.path.dirname(video_path), "resized_" + video_name)
    out = cv2.VideoWriter(output_video_path, fourcc, ref_fps, (ref_width, ref_height))

    for i in tqdm(range(num_frames)):
        ret, frame = cap.read()
        if not ret:
            break

        # Resize the frame
        resized_frame = cv2.resize(frame, (ref_width, ref_height))
        out.write(resized_frame)

    # Cleanup
    cap.release()
    out.release()
    print(f"Resized video saved to {output_video_path}")

def archived_video_threshold(video_path, threshold_percent=0.85, init_errode=2, n_binaries=3):

    # Open the input video file
    cap = cv2.VideoCapture(video_path)
    if not cap.isOpened():
        print("Error: Could not open input video file.")
        exit()

    # Get the video properties
    fps = int(cap.get(cv2.CAP_PROP_FPS))
    width = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
    height = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))

    # Define the codec and create VideoWriter object
    fourcc = cv2.VideoWriter_fourcc(*'MJPG')
    video_name = os.path.split(video_path)[1]
    output_video_path = os.path.join(os.path.dirname(video_path), "threshold_" + video_name)
    out = cv2.VideoWriter(output_video_path, fourcc, fps, (width, height))

    # Process the video frame by frame
    while True:
        ret, frame = cap.read()

        if not ret:
            break

        # Remove 4th dimension (colors)
        frame = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)

        # Apply your filters/transformations to the frame here
        background_substracted_frame = np.abs(frame - background)
        blurred_bsf = gaussian_blur(background_substracted_frame, 3)
        uint8_blurred_bsf = np.uint8(blurred_bsf)
        uint8_blurred_bsf = cv2.cvtColor(uint8_blurred_bsf, cv2.COLOR_GRAY2BGR)
        out.write(uint8_blurred_bsf)

        # Write the modified frame to the output video
        out.write(uint8_blurred_bsf)

    # Release the video capture and writer objects
    cap.release()
    out.release()
    print(f"Output video saved to {output_video_path}")

    return None

def archived_compute_centroids(mask_stack_path):
    # Open the input video file
    cap = cv2.VideoCapture(mask_stack_path)
    if not cap.isOpened():
        raise ValueError("Error: Could not open input video file.")

    # Get the number of frames in the video
    n_frames = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))

    # Initialize an array to store the centroid coordinates
    centroids = np.zeros((n_frames, 2), dtype=np.int16)

    # Process the video frame by frame
    for i in range(n_frames):
        ret, frame = cap.read()

        if not ret:
            raise ValueError(f"Error: Could not read frame {i}.")

        # Convert frame to grayscale if it is in color
        if len(frame.shape) == 3:
            frame = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)

        # Threshold the frame to get the binary mask
        _, mask = cv2.threshold(frame, 127, 255, cv2.THRESH_BINARY)

        # Find the contours in the binary mask
        contours, _ = cv2.findContours(mask, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)

        # Compute the centroid for the largest contour
        if contours:
            largest_contour = max(contours, key=cv2.contourArea)
            M = cv2.moments(largest_contour)
            cx = int(M['m10'] / M['m00'])
            cy = int(M['m01'] / M['m00'])
            centroids[i, :] = [cx, cy]

    # Release the video capture object
    cap.release()

    return centroids


def archived_crop_video_and_depth_first_one(behavior_folder):
    video_path = os.path.join(behavior_folder, "video.avi")
    depth_path = os.path.join(behavior_folder, "depth.avi")
    # Open the input video file
    cap = cv2.VideoCapture(video_path)
    if not cap.isOpened():
        print("Error: Could not open input video file.")
        return

    # Read the first frame
    ret, first_frame = cap.read()
    if not ret:
        print("Error: Could not read the first frame.")
        cap.release()
        return

    def select_and_crop(image):
        # Create a window to display the image
        cv2.namedWindow('Select ROI')
        cv2.imshow('Select ROI', image)

        # Use OpenCV's selectROI function to select a rectangular region
        r = cv2.selectROI('Select ROI', image)

        # Crop the image using the coordinates of the selected rectangle
        cropped_image = image[int(r[1]):int(r[1] + r[3]), int(r[0]):int(r[0] + r[2])]

        # Close the ROI selection window
        cv2.waitKey(10)
        cv2.destroyAllWindows()
        cv2.waitKey(10)

        return cropped_image, r

    # Select and crop the first frame
    cropped_first_frame, roi = select_and_crop(first_frame)

    # Define the codec and create VideoWriter object
    fourcc = cv2.VideoWriter_fourcc(*'MJPG')
    video_name = os.path.split(video_path)[1]
    output_video_path = os.path.join(os.path.dirname(video_path), "cropped_" + video_name)
    out = cv2.VideoWriter(output_video_path, fourcc, 31.0, (cropped_first_frame.shape[1], cropped_first_frame.shape[0]))

    # Write the first cropped frame
    out.write(cropped_first_frame)

    # Get the total number of frames for the video
    total_frames = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))

    # Process and write the frames with tqdm
    for _ in tqdm(range(total_frames), desc="Processing video"):
        ret, frame = cap.read()

        if not ret:
            break
        cropped_frame = frame[int(roi[1]):int(roi[1] + roi[3]), int(roi[0]):int(roi[0] + roi[2])]
        out.write(cropped_frame)

    # Release the video capture and writer objects
    cap.release()
    out.release()
    print(f"Output video saved to {output_video_path}")

    cap2 = cv2.VideoCapture(depth_path)
    if not cap2.isOpened():
        print("Error: Could not open depth file.")
        return

    # Define the codec and create VideoWriter object
    fourcc = cv2.VideoWriter_fourcc(*'MJPG')
    depth_name = os.path.split(depth_path)[1]
    output_depth_path = os.path.join(os.path.dirname(depth_path), "cropped_" + depth_name)
    out2 = cv2.VideoWriter(output_depth_path, fourcc, 31.0,
                           (cropped_first_frame.shape[1], cropped_first_frame.shape[0]))

    #     affine_matrix = np.array([[1.10195914, 0.        ], [0.        , 1.10195914]])
    #     offset = np.array([-41.77996883, -19.86813022])

    affine_matrix = np.array([[0.92, 0.], [0., 0.92]])
    offset = np.array([13., 34.])
    affine_matrix_cv2 = np.hstack([affine_matrix, offset.reshape(-1, 1)])

    # Get the total number of frames for the depth video
    total_frames_depth = int(cap2.get(cv2.CAP_PROP_FRAME_COUNT))

    # Process and write the depth frames with tqdm
    for _ in tqdm(range(total_frames_depth), desc="Processing depth"):
        ret, frame = cap2.read()
        if not ret:
            break
        trsfmd_frame = cv2.warpAffine(frame, affine_matrix_cv2, (frame.shape[1], frame.shape[0]))
        cropped_frame = trsfmd_frame[int(roi[1]):int(roi[1] + roi[3]), int(roi[0]):int(roi[0] + roi[2])]
        out2.write(cropped_frame)

    # Release the video capture and writer objects
    cap2.release()
    out2.release()
    print(f"Output depth saved to {output_depth_path}")

    return None
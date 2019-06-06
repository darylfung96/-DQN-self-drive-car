from collections import deque
import numpy as np

from skimage import transform
from skimage.color import rgb2gray

stack_frame_size = 4


def preprocess_frame(frame):
    gray_frame = rgb2gray(frame)

    # reduce complexity
    # Crop the screen (remove the part below the player)
    # [Up: Down, Left: right]
    cropped_frame = gray_frame[8:-12, 4:-12]

    normalized_frame = cropped_frame / 255.0
    preprocessed_frame = transform.resize(normalized_frame, [110, 84])
    return preprocessed_frame


def stack_frames(stacked_frames, new_frame, is_done):
    if stacked_frames is None:
        stacked_frames = [np.zeros((110, 84)) for _ in range(stack_frame_size)]

    new_frame = preprocess_frame(new_frame)

    if is_done:
        stacked_frames = [np.zeros((110, 84)) for _ in range(stack_frame_size)]
        stacked_frames = deque(stacked_frames, maxlen=stack_frame_size)

        # generate stacked frames
        for _ in range(stack_frame_size):
            stacked_frames.append(new_frame)
    else:
        stacked_frames.append(new_frame)

    stacked_state = np.stack(stacked_frames, axis=0).astype(np.float32)
    return stacked_state, stacked_frames

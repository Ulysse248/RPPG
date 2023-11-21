import cv2
import numpy as np

def boxidx(idx, val):
    rg = 3
    return list(range(idx-rg, idx+rg)), list(range(val-rg, val+rg))

def output(signal, video_path):
    cap = cv2.VideoCapture(video_path)
    try:
        frames = int(cap.get(cv2.cv.CV_CAP_PROP_FRAME_COUNT))
        width  = int(cap.get(cv2.cv.CV_CAP_PROP_FRAME_WIDTH))
        height = int(cap.get(cv2.cv.CV_CAP_PROP_FRAME_HEIGHT))
    except AttributeError:
        frames = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
        width  = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
        height = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
        fps = cap.get(cv2.CAP_PROP_FPS)

    peak_to_peak = max(signal) - min(signal)
    mean_sig = np.mean(signal)
    signal = np.array(signal)
    l=200
    new_signal = l/peak_to_peak*(signal - mean_sig) + height//2
    flag = True
    signal_on_frames = []
    idx = 3
    previdx = []
    preval = []

    new_frames = []
    while flag:
        _, frame = cap.read()
        new_frame = frame
        boxid, boxval = boxidx(idx, int(new_signal[idx]))
        previdx += boxid
        preval += boxval

        new_frame[preval, previdx] = (0, 0, 255)
        new_frames.append(new_frame)

        if idx==len(signal) - 1:
            flag = False
        idx+=1
    print(len(new_frames))
    codec = cv2.VideoWriter_fourcc(*'mp4v')  # Choose the codec (codec may vary based on your system)
    frame_size = (new_frames[0].shape[1], new_frames[0].shape[0])  # Width x Height (assuming frames are in H x W x C format)

    # Create VideoWriter object to write the frames into a video file
    out = cv2.VideoWriter('output_video.mp4', codec, fps, frame_size)

    # Write frames to the video file
    for frame in new_frames:
        out.write(frame)

    # Release the VideoWriter and close the file
    out.release()

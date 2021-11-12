from ast import parse
import cv2
from glob import glob
from os import path
import tqdm
import numpy as np
from multiprocessing import Pool
import multiprocessing
import argparse
import h5py

INPUT_FOLDER = path.abspath("../DeepStab")

def ExtractSURF(v_path, out_path):

    surf = cv2.xfeatures2d.SURF_create()
    
    cap = cv2.VideoCapture(v_path)
    if not cap.isOpened():
        print(f"Fail to open {v_path}")
        return False

    dataset = h5py.File(out_path, "w")
    dataset.create_group('keypoints')
    dataset.create_group('descriptors')

    frame_count = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))

    pbar = tqdm.tqdm(total=frame_count, leave=False)
    w, h = 512, 288
    i = 0

    while True:
        ret, frame = cap.read()
        if not ret:
            break

        frame = cv2.resize(frame, (w, h))
        frame = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)

        keypoints, descriptors = surf.detectAndCompute(frame, None)
        locations = np.array([k.pt for k in keypoints])
        # import pdb; pdb.set_trace()
        dataset['keypoints'].create_dataset(str(i), data = locations, compression="gzip", compression_opts=9, dtype=np.float32)
        dataset['descriptors'].create_dataset(str(i), data = descriptors, compression="gzip", compression_opts=9, dtype=np.float32)

        i+=1

        pbar.update()
    
    cap.release()
    pbar.close()

    return True

def parse():
    parser = argparse.ArgumentParser()
    parser.add_argument("--input",  "-i", default=INPUT_FOLDER, required=False)
    return parser.parse_args()

if __name__ == "__main__":

    args = parse()

    all_videos = glob(path.join(args.input, "**/*.avi"), recursive=True)
    print(f"Found {len(all_videos)} videos")


    for v_path in all_videos:
        v_name =  v_path.split("/")[-1]
        out_path = v_path.replace('.avi', "_surf.h5")
        if not path.isfile(out_path):
            tqdm.tqdm.write(f"Extracting video {v_name}")
            ExtractSURF(v_path, out_path)
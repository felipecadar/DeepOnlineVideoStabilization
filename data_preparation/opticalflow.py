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

def CalcOF(v_path):
    try:
        # In case you're using Pool
        pos = multiprocessing.current_process()._identity[0]
    except:
        pos = 0
    
    cap = cv2.VideoCapture(v_path)

    if not cap.isOpened():
        print(f"Fail to open {v_path}")
        return False

    frame_count = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))

    pbar = tqdm.tqdm(total=frame_count, leave=False, position=pos)
    w, h = 512, 288
    ret, frame = cap.read()
    frame = cv2.resize(frame, (w, h))
    prvs = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)

    out_path = v_path.split(".")[0] + "_OF.h5"
    dataset = h5py.File(out_path, "w")
    dataset.create_group('flow')

    optical_flow = cv2.optflow.DualTVL1OpticalFlow_create()
    
    i = 0
    while ret:
        ret, frame = cap.read()
        if not ret:
            break

        frame = cv2.resize(frame, (w, h))
        next = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)

        flow = optical_flow.calc(prvs, next, None)
        dataset['flow'].create_dataset(str(i), data = flow, compression="gzip", compression_opts=9, dtype=np.float32)


        prvs = next
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
    input_folder = path.abspath(args.input)
    if not path.isdir(input_folder):
        print("Fail to find input folder", input_folder)
        exit()

    all_videos = glob(path.join(args.input, "**/*.avi"), recursive=True)
    print(f"Found {len(all_videos)} videos")

    if not args.pool:
        for v_path in all_videos:
            v_name =  path.basename(v_path)
            out_path = v_path.replace('.avi', "_OF.h5")
            if not path.isfile(out_path):
                tqdm.tqdm.write(f"Extracting video {v_name}")
                CalcOF(v_path, out_path)
            else:
                print(f"Found {out_path}")


            

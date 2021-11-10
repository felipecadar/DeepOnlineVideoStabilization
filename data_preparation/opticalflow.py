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

    out_path = v_path.split(".")[0] + "_OF"
    dataset = h5py.File(out_path, "w")
    dataset.create_group('imgs')

    optical_flow = cv2.optflow.DualTVL1OpticalFlow_create()
    
    i = 0
    while ret:
        ret, frame = cap.read()
        if not ret:
            break

        frame = cv2.resize(frame, (w, h))
        next = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)

        flow = optical_flow.calc(prvs, next, None)
        dataset['imgs'].create_dataset(str(i), data = flow, compression="gzip", compression_opts=9, dtype=np.float32)


        prvs = next
        i+=1
        pbar.update()
    
    cap.release()
    pbar.close()

    return True

def parse():
    parser = argparse.ArgumentParser()
    parser.add_argument("--input",  "-i", default=INPUT_FOLDER, required=False)
    parser.add_argument("--pool",  "-p", default=False, action="store_true", required=False)
    parser.add_argument("--n-proc", type=int, default=-1, required=False)
    return parser.parse_args()

if __name__ == "__main__":

    args = parse()

    all_videos = glob(path.join(args.input, "**/*.avi"), recursive=True)
    print(f"Found {len(all_videos)} videos")

    if not args.pool:
        for v_path in all_videos:
            v_name =  v_path.split("/")[-1]
            npz = v_path.split(".")[0] + "_OF.npz"
            if not path.isfile(npz):
                tqdm.tqdm.write(f"Extracting video {v_name}")
                CalcOF(v_path)
            else:
                print(f"Found {npz}")
    else:
        pool_args = []
        for v_path in all_videos:
            v_path = path.abspath(v_path)
            v_name =  v_path.split("/")[-1]
            npz = v_path.split(".")[0] + "_OF.npz"
            if not path.isfile(npz):
                pool_args.append(v_path)
        
        if args.n_proc == -1:
            proc = multiprocessing.cpu_count()
        else:
            proc = args.n_proc

        with Pool(proc) as p:
            res = p.map(CalcOF, pool_args)

        for x in res:
            assert(x)

            

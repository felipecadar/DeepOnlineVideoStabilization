from ast import parse
import cv2
from glob import glob
from os import path
import tqdm
import numpy as np
from multiprocessing import Pool
import multiprocessing
import argparse
INPUT_FOLDER = path.abspath("../DeepStab")

def CalcOF(v_path):
    try:
        # In case you're using Pool
        pos = multiprocessing.current_process()._identity[0]
    except:
        pos = 0
    
    use_memmap = False
    optical_flow = cv2.optflow.DualTVL1OpticalFlow_create()
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

    if use_memmap:
        of_x = np.memmap("of_x.dat", dtype='float32', mode='w+', shape=(h, w, frame_count))
        of_y = np.memmap("of_y.dat", dtype='float32', mode='w+', shape=(h, w, frame_count))
    else:
        of_x = np.zeros((h, w, frame_count), dtype=np.float32)
        of_y = np.zeros((h, w, frame_count), dtype=np.float32)

    i = 0
    while ret:
        ret, frame = cap.read()
        if not ret:
            break

        frame = cv2.resize(frame, (w, h))
        next = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)

        flow = optical_flow.calc(prvs, next, None)
        of_x[:,:,i] = flow[:,:,0]
        of_y[:,:,i] = flow[:,:,1]
        
        if use_memmap:
            of_x.flush()
            of_y.flush()

        prvs = next

        pbar.update()

    np.savez_compressed(v_path.split(".")[0] + "_OF", of_x = of_x, of_y=of_y)
    
    cap.release()
    pbar.close()

    return True

def parse():
    parser = argparse.ArgumentParser()
    parser.add_argument("--input",  "-i", default=INPUT_FOLDER, required=False)
    parser.add_argument("--pool",  "-p", default=False, action="store_true", required=False)
    parser.add_argument("--n-proc", "-p", type=int, default=-1, required=False)
    return parser.parse_args()

if __name__ == "__main__":

    args = parse()

    all_videos = glob(path.join(args.input, "**/*.avi"), recursive=True)
    print(f"Found {len(all_videos)} videos")

    if args.pool:
        for v_path in all_videos:
            v_name =  v_path.split("/")[-1]
            npz = v_path.split(".")[0] + "_OF.npz"
            if not path.isfile(npz):
                tqdm.tqdm.write(f"Extracting video {v_name}")
                CalcOF(v_path)
    else:
        pool_args = []
        for v_path in all_videos:
            v_name =  v_path.split("/")[-1]
            npz = v_path.split(".")[0] + "_OF.npz"
            if not path.isfile(npz):
                pool_args.append(v_path)
        
        if args.p == -1:
            proc = multiprocessing.cpu_count()
        else:
            proc = args.p

        with Pool(proc) as p:
            res = p.map(CalcOF, args)

        for x in res:
            assert(x)

            

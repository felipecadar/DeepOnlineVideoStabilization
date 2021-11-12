from ast import parse
import pdb
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
DEBUG = False
def isInside(ids, kp, p0, p1):
    filter_id = ((kp >= p0).all(axis=1) & (kp <= p1).all(axis=1))
    return filter_id

def filter_matches(kp1, kp2, matches, ratio = 0.85):
    mkp1, mkp2 = [], []
    id_mkp1, id_mkp2 = [], []
    for m in matches:
        if len(m) == 2 and m[0].distance < m[1].distance * ratio:
            m = m[0]

            id_mkp1.append( m.queryIdx )
            id_mkp2.append( m.trainIdx )

    

    return np.array(id_mkp1), np.array(id_mkp2)

def MatchSURF(stable_path, unstable_path, out_path):   

    stable_surf = h5py.File(stable_path, "r")
    unstable_surf = h5py.File(unstable_path, "r")

    dataset = h5py.File(out_path, "w")
    dataset.create_group('matches')

    frame_count = len(stable_surf['descriptors'])
    w, h = 512, 288

    matcher = cv2.BFMatcher(cv2.NORM_L2)


    for frame in tqdm.tqdm(range(frame_count)):
        stable_descs = stable_surf['descriptors'][str(frame)][()]
        stable_kps = stable_surf['keypoints'][str(frame)][()]

        unstable_descs = unstable_surf['descriptors'][str(frame)][()]
        unstable_kps = unstable_surf['keypoints'][str(frame)][()]

        raw_matches = matcher.knnMatch(stable_descs, unstable_descs, k = 2)
        id_mkp1, id_mkp2 = filter_matches(stable_kps, unstable_kps, raw_matches)

        RANSAC_matches1 = []
        RANSAC_matches2 = []

        n_grid = 3
        xv, yv = np.meshgrid(np.linspace(0, w-1, n_grid, dtype=int),  np.linspace(0, h-1, n_grid, dtype=int))
        for i in range(0, n_grid-1):
            for j in range(0, n_grid-1):
                p0 = np.array([xv[i,j], yv[i,j]])
                p1 = np.array([xv[i+1,j+1], yv[i+1,j+1]])

                filter_ids = isInside(id_mkp1, stable_kps[id_mkp1], p0, p1)

                valid_mkp1 = id_mkp1[filter_ids]
                valid_mkp2 = id_mkp2[filter_ids]
                
                if len(valid_mkp1) < 4:
                    tqdm.tqdm.write('Cant compute ransac with less then 4 Kps')
                    continue

                if DEBUG:
                    print('Before RANSAC')
                    print(valid_mkp1.shape)
                    print(valid_mkp2.shape)
                
                if DEBUG:
                    print(h, w, 3)
                    img = np.zeros([h, w, 3], dtype=np.uint8)
                    for c in id_mkp1:
                        cv2.circle(img, stable_kps[c].astype(int), 3, [0,0,255], -1)
                    
                    cv2.imshow('', img)
                    cv2.waitKey()


                _, status = cv2.findHomography(stable_kps[valid_mkp1], unstable_kps[valid_mkp2], cv2.RANSAC,5.0)
                    
                valid_mkp1 = valid_mkp1[status[:, 0].astype(bool)]
                valid_mkp2 = valid_mkp2[status[:, 0].astype(bool)]
                if DEBUG:
                    print('After RANSAC')
                    print(valid_mkp1.shape)
                    print(valid_mkp2.shape)

                RANSAC_matches1.extend([x for x in valid_mkp1])
                RANSAC_matches2.extend([x for x in valid_mkp2])
        

        RANSAC_matches1 = np.array(RANSAC_matches1)
        RANSAC_matches2 = np.array(RANSAC_matches2)

        final_matches = np.hstack([RANSAC_matches1, RANSAC_matches2])

        dataset.create_dataset(str(frame),data=final_matches, compression="gzip", compression_opts=9, dtype=int )

        if DEBUG:
            tqdm.tqdm.write('Total Kps: {}'.format( RANSAC_matches1.shape))

        

        # import pdb; pdb.set_trace()



    return True

def parse():
    parser = argparse.ArgumentParser()
    parser.add_argument("--input",  "-i", default=INPUT_FOLDER, required=False)
    return parser.parse_args()

if __name__ == "__main__":

    args = parse()

    all_videos = sorted(glob(path.join(args.input, "stable/*.avi"), recursive=True))
    print(f"Found {len(all_videos)} videos")


    for v_path in all_videos:
        v_name =  v_path.split("/")[-1].split('.')[0]

        path_desc_stable = v_path.replace('.avi', "_surf.h5")
        path_desc_unstable = v_path.replace('.avi', "_surf.h5").replace('/stable/', '/unstable/')

        out_path = v_path.replace('.avi', "_matches.h5")
        if not path.isfile(out_path):
            tqdm.tqdm.write(f"Matching video {v_name}")
            MatchSURF(path_desc_stable, path_desc_unstable, out_path)

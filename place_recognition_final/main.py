import os
import argparse
import numpy as np
import sys
import cv2
np.set_printoptions(precision=4)

import matplotlib.pyplot as plt
import utils.UtilsDataset as DtUtils
import utils.UtilsVisualization as VsUtils

from tqdm import tqdm
from utils.DescriptorDistanceManager import *
from pathlib import Path

parser = argparse.ArgumentParser(description='Loop Detection Evaluation')

class MainParser():
    def __init__(self):
        FILE = Path(__file__).resolve()             
        ROOT = FILE.parents[1]                              
        ROOT = Path(os.path.relpath(ROOT, Path.cwd()))
        self.parser = argparse.ArgumentParser(description= "This is a initial option for find matching pair",
                                              fromfile_prefix_chars='@')
        self.root = ROOT
        self.parser.convert_arg_line_to_args = self.convert_arg_line_to_args
        self.initialize()
        
    def initialize(self):
        '''
        Please check the holoocean.txt
        '''
        self.parser.add_argument('--cosine_similarity_threshold',             type=float,   default=0.67, help = 'cosine_similarity_threshold')
        self.parser.add_argument('--img_dir_all',                           type=str,   default='/home/cheol/carla_ws/src/image_matching_test_FGI/60m_100m_test/60m_100m_imgs', help = 'Folder path of entire Image')
        self.parser.add_argument('--img_dir_query',                           type=str,   default='/home/cheol/carla_ws/src/image_matching_test_FGI/60m_100m_test/similar_ts_image_60_3', help = 'Folder path of query Image')
        self.parser.add_argument('--img_dir_database',                           type=str,   default='/home/cheol/carla_ws/src/image_matching_test_FGI/60m_100m_test/similar_ts_image_100_3', help = 'Folder path of database Image')
        self.parser.add_argument('--pose_dir',                                type=str,   default='/home/cheol/carla_ws/src/image_matching_test_FGI/60m_100m_test/gt_poses/60_100_3.csv', help = 'Path of Pose-csv')
        self.parser.add_argument('--exclude_recent_nodes',                    type=int,   default=30, help = 'Recently visited nodes number')     
        self.parser.add_argument('--desc_vec_dir',                            type=str,   default='/home/cheol/carla_ws/src/image_matching_test_NetVLAD/60_100_altitude_NetVlad/place_recognition/60_100_Desc_numpy.npy', help = 'Numpy array of Descriptor Vector')     

        #! global pose가 아니라 Local pose라서 수정해야 함
        self.parser.add_argument('--csvx',                                    type=int,   default=1, help = 'Local x in pose-csv')               
        self.parser.add_argument('--csvy',                                    type=int,   default=2, help = 'Local y in pose-csv')
        self.parser.add_argument('--csvz',                                     type=int,   default=3, help = 'Local z in pose-csv')              

    def convert_arg_line_to_args(self, arg_line):
        for arg in arg_line.split():
            if not arg.strip():
                continue
            yield arg

    def parse(self):
        if sys.argv.__len__() == 2:
            arg_filename_with_prefix = '@' + sys.argv[1]
            args = self.parser.parse_args([arg_filename_with_prefix])
        else:
            args = self.parser.parse_args()
        return args

src_node = []
dst_node = []
dist_node = []



if __name__ == '__main__':
    parser = MainParser()
    args = parser.parse()
    
    # img_manager = DtUtils.ImgDirManager(args.img_dir)
    img_manager_all = DtUtils.ImgDirManager(args.img_dir_all)
    img_paths_all = img_manager_all.img_fullpaths
    
    img_manager_query = DtUtils.ImgDirManager(args.img_dir_query)
    img_paths_qeury = img_manager_query.img_fullpaths
    
    img_manager_database = DtUtils.ImgDirManager(args.img_dir_database)
    img_paths_database = img_manager_database.img_fullpaths
    
    desc_vec_path = args.desc_vec_dir 
    

    #!=====================================================================================
    DCM = DescriptorManager(args.cosine_similarity_threshold)

    pose_manager = DtUtils.PoseDirManager(args.pose_dir)
    pose = pose_manager.getPose(args.csvx, args.csvy, args.csvz)
    
    src_node = []
    dst_node = []
    dist_node = []
    desc_vec_all = np.load(desc_vec_path)
    query_size = len(img_paths_qeury)
    
    cnt = 0
    for for_idx, desc_vec in tqdm(enumerate(desc_vec_all), total=desc_vec_all.shape[0], mininterval=1):  
        DCM.addNode(node_idx = for_idx, desc_vec = desc_vec)
    
    for idx in range(query_size):
        loop_idx, max_sim_score = DCM.detectLoop(args.exclude_recent_nodes, idx, query_size) 
        print("loop_idx :::: ", loop_idx)
        
        if(loop_idx == None):
            pass
        else:
            src_node.append(idx)
            dst_node.append(loop_idx + query_size)        # multi-session 이기 때문
            dist_node.append(max_sim_score)
            
                
    total_loop_num = len(src_node)
    print(f"total loop num : {total_loop_num}")

    src_node = np.array(src_node)[:,None]
    dst_node = np.array(dst_node)[:,None]
    
    loop = np.concatenate((src_node, dst_node), axis=1)
    viz = VsUtils.VizTrajectory(loop, pose)
    viz.viz2D()
    viz.viz3D()
        
    plt.show()
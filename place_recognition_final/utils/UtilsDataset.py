import os
import csv
import numpy as np
import natsort
    
class ImgDirManager:
    def __init__(self, img_dir):
        self.img_dir = os.path.join(img_dir)                      
        self.imgfile_list = os.listdir(self.img_dir)                                              
        self.imgfile_list = natsort.natsorted(self.imgfile_list)                                                                   
        self.img_fullpaths = [os.path.join(self.img_dir, name) for name in self.imgfile_list]    
        self.num_imgs = len(self.imgfile_list)                                                    
        
class PoseDirManager:
    def __init__(self, pose_base_dir):
        self.pose_path = pose_base_dir
    
    def getPose(self, csvx, csvy, csvz):
        f = open(self.pose_path, 'r', encoding='utf-8')
        rdr = csv.reader(f)
        x, y, z = [], [], []
        
        # Time elevation trajectory
        for line in rdr:
            x.append(float(line[csvx]))
            y.append(float(line[csvy]))
            z.append(float(line[csvz]))
            
        pose = np.vstack([x,y,z])
        pose = np.transpose(pose)
        
        return pose
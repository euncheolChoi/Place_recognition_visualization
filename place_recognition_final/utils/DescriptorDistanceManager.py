import numpy as np
import cv2
import sys
from scipy import spatial
from numpy import dot
from numpy.linalg import norm

np.set_printoptions(precision=4)

# top_idx, max_cos_sim = cosine_similarity(sonarcontexts_history[idx], self.sonarcontexts_100_meter, self.cosine_similarity)    
def cosine_similarity(desc_curr, desc_vec_all, threshold):
    sim_score_list = []
    
    if desc_vec_all == None:
        print("desc_curr : ", desc_curr)
        return None, None
    else:
        for idx, desc_test in enumerate(desc_vec_all):
            if desc_curr is not None and desc_test is not None:
                
                # test_vec_1 = [0, 1, 1, 1]
                # test_vec_2 = [1, 0, 1, 1]
                # sim_score_test = np.dot(test_vec_1, test_vec_2) / (np.linalg.norm(test_vec_1) * np.linalg.norm(test_vec_2))
                # print("test vec score ?????????????????????????????????? : ", sim_score_test )
                
                sim_score = np.dot(desc_curr, desc_test) / (np.linalg.norm(desc_curr) * np.linalg.norm(desc_test))
                sim_score_list.append(sim_score) 
           

    top_idx = sim_score_list.index(max(sim_score_list))
    max_sim_score = sim_score_list[top_idx]
    if max_sim_score > threshold:
        return top_idx, max_sim_score
    else:
        return None, None
    
    

    return top_idx , max_sim_score

class DescriptorManager:
    
    # def __init__(self, inlier_threshold, cosine_similarity):
    def __init__(self, cosine_similarity):
        # self.inlier_threshold = inlier_threshold
        self.ENOUGH_LARGE = 600                            
        self.sonarcontexts = [None] * self.ENOUGH_LARGE     
        self.sonarcontexts_60_meter = [None] * self.ENOUGH_LARGE     
        self.sonarcontexts_100_meter = [None] * self.ENOUGH_LARGE     
        self.curr_node_idx = 0
        self.cosine_similarity = cosine_similarity

        
    def addNode(self, node_idx, desc_vec):        # sc는 이미지가 아니라 디스크립터 벡터 
        self.curr_node_idx = node_idx
        self.sonarcontexts[node_idx] = desc_vec
        
        # if 0 <= node_idx <= 215:
        #     self.sonarcontexts_60_meter[node_idx] = desc_vec
        
        # if 216 <= node_idx <= 563:
        #     self.sonarcontexts_100_meter[node_idx] = desc_vec

    def detectLoop(self, exclude_recent_nodes, curr_idx, query_size):
        valid_recent_node_idx = self.curr_node_idx - exclude_recent_nodes   # 너무 가까운 씬은 무조건 매치로 잡힐것이므로, 몇개의 프레임을 제외할것인가?
        sonarcontexts_history = self.sonarcontexts   
        top_idx, max_cos_sim = cosine_similarity(sonarcontexts_history[curr_idx], sonarcontexts_history[query_size:], self.cosine_similarity)              
        
        if (top_idx == None):
            return None, None
        

        if(max_cos_sim >= self.cosine_similarity):     
            # print("cosine_similarity : " , max_cos_sim)
            # print("Top index" , top_idx)
            # print("current_node_idx : ", self.curr_node_idx)
            # print("max_sim_score : ", max_cos_sim)
            
            #! return top_idx + 217, max_cos_sim -> 이게 아님 !! 왜냐하면, 애초에 self.sonarcontexts_100_meter[node_idx] = desc_vec에서 node_idx가 현재 노드의 인덱스이니까! 
            return top_idx , max_cos_sim
        else:
            return None, None
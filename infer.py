import numpy as np
from PIL import Image
import os
import glob 
import matplotlib.pyplot as plt

import cv2
import scipy.spatial.distance as dis
from run import Scoring
scr = Scoring()

path = '/home/dsoft/Music/me/project/tf-pose-estimation/images/donwline_Address.jpeg'
path1 = '/home/dsoft/Music/me/project/tf-pose-estimation/images/faceon_Toe-up.jpeg'
img_list = []
path_list = '/home/dsoft/Music/me/project/tf-pose-estimation/image_test_copy (copy)'

class EventScoring():
    def __init__(self):
        pass

    # def set_config(self, id_event):
    #     """id_event is number of event what is series of golf swing
    #     Args:
    #         id_event (int): number of events

    #     """
    #     if id_event==0:
    #         x_right = [0.4722222222222222, 0.5, 0.5055555555555555, 0.4666666666666667, 0.45555555555555555, 0.48333333333333334, 0.46111111111111114, 0.45, 0.5388888888888889, 0.55, 0.6055555555555555, 0.4777777777777778, 0.4888888888888889]
    #         y_right = [0.3388888888888889, 0.37222222222222223, 0.37222222222222223, 0.4166666666666667, 0.4, 0.5555555555555556, 0.6666666666666666, 0.7833333333333333, 0.5555555555555556, 0.6777777777777778, 0.7833333333333333, 0.3333333333333333, 0.3388888888888889]
    #     if id_event==1:
    #         x_right = [0.4888888888888889, 0.5166666666666667, 0.48333333333333334, 0.4388888888888889, 0.42777777777777776, 0.55, 0.5166666666666667, 0.45, 0.5, 0.5055555555555555, 0.45, 0.5611111111111111, 0.6055555555555555, 0.6111111111111112, 0.48333333333333334, 0.5, 0.5222222222222223]
    #         y_right = [0.40555555555555556, 0.4222222222222222, 0.43333333333333335, 0.4666666666666667, 0.4666666666666667, 0.42777777777777776, 0.4666666666666667, 0.4777777777777778, 0.5611111111111111, 0.6777777777777778, 0.7777777777777778, 0.5555555555555556, 0.6666666666666666, 0.7777777777777778, 0.4, 0.39444444444444443, 0.3888888888888889]
    #     if id_event==2:
    #         x_right = [0.4777777777777778, 0.49444444444444446, 0.5055555555555555, 0.4388888888888889, 0.37222222222222223, 0.4777777777777778, 0.45555555555555555, 0.45, 0.5333333333333333, 0.55, 0.6055555555555555, 0.4777777777777778, 0.48333333333333334, 0.4888888888888889]
    #         y_right = [0.3888888888888889, 0.4166666666666667, 0.4222222222222222, 0.4388888888888889, 0.42777777777777776, 0.5666666666666667, 0.6666666666666666, 0.7833333333333333, 0.5611111111111111, 0.6777777777777778, 0.7888888888888889, 0.37222222222222223, 0.37777777777777777, 0.38333333333333336]
    #     if id_event==3:
    #         x_right = [0.48333333333333334, 0.5, 0.46111111111111114, 0.4388888888888889, 0.38333333333333336, 0.5333333333333333, 0.5055555555555555, 0.45555555555555555, 0.4722222222222222, 0.45555555555555555, 0.45, 0.5333333333333333, 0.5611111111111111, 0.6055555555555555, 0.4777777777777778, 0.49444444444444446, 0.5222222222222223]
    #         y_right = [0.4, 0.4166666666666667, 0.4111111111111111, 0.4722222222222222, 0.5444444444444444, 0.42777777777777776, 0.4777777777777778, 0.5111111111111111, 0.5611111111111111, 0.6611111111111111, 0.7833333333333333, 0.5555555555555556, 0.6722222222222223, 0.7833333333333333, 0.38333333333333336, 0.3888888888888889, 0.38333333333333336]
    #     if id_event==4:
    #         x_right = [0.4888888888888889, 0.49444444444444446, 0.46111111111111114, 0.48333333333333334, 0.5388888888888889, 0.5333333333333333, 0.5611111111111111, 0.5666666666666667, 0.5111111111111111, 0.5722222222222222, 0.6055555555555555, 0.55, 0.5333333333333333, 0.46111111111111114, 0.4777777777777778, 0.49444444444444446, 0.4777777777777778, 0.5]
    #         y_right = [0.4, 0.40555555555555556, 0.42777777777777776, 0.49444444444444446, 0.5555555555555556, 0.39444444444444443, 0.4444444444444444, 0.5333333333333333, 0.5444444444444444, 0.6388888888888888, 0.7722222222222223, 0.5333333333333333, 0.6611111111111111, 0.7722222222222223, 0.39444444444444443, 0.38333333333333336, 0.39444444444444443, 0.37777777777777777]
    #     if id_event==5:
    #         x_right = [0.5055555555555555, 0.5277777777777778, 0.4777777777777778, 0.49444444444444446, 0.5277777777777778, 0.5722222222222222, 0.5611111111111111, 0.55, 0.5, 0.4777777777777778, 0.45555555555555555, 0.5611111111111111, 0.5833333333333334, 0.6055555555555555, 0.5, 0.5111111111111111, 0.4888888888888889, 0.5333333333333333]
    #         y_right = [0.4, 0.4166666666666667, 0.42777777777777776, 0.5, 0.5833333333333334, 0.4111111111111111, 0.49444444444444446, 0.5777777777777777, 0.5555555555555556, 0.6666666666666666, 0.7833333333333333, 0.5555555555555556, 0.6611111111111111, 0.7833333333333333, 0.39444444444444443, 0.3888888888888889, 0.3888888888888889, 0.37222222222222223]
    #     if id_event==6:
    #         x_right = [0.4722222222222222, 0.4777777777777778, 0.45555555555555555, 0.4777777777777778, 0.5666666666666667, 0.5, 0.5666666666666667, 0.6333333333333333, 0.5222222222222223, 0.5166666666666667, 0.45, 0.5722222222222222, 0.5833333333333334, 0.6055555555555555, 0.4777777777777778, 0.4722222222222222, 0.4722222222222222]
    #         y_right = [0.3888888888888889, 0.39444444444444443, 0.4111111111111111, 0.4722222222222222, 0.4777777777777778, 0.38333333333333336, 0.43333333333333335, 0.4722222222222222, 0.5444444444444444, 0.6555555555555556, 0.7722222222222223, 0.5333333333333333, 0.65, 0.7722222222222223, 0.38333333333333336, 0.3611111111111111, 0.3611111111111111]
    #     elif id_event==7:
    #         x_right = [0.5611111111111111, 0.6, 0.5222222222222223, 0.45555555555555555, 0.4777777777777778, 0.5888888888888889, 0.55, 0.4444444444444444, 0.5555555555555556, 0.5611111111111111, 0.5944444444444444, 0.5833333333333334, 0.5722222222222222, 0.5444444444444444]
    #         y_right = [0.37222222222222223, 0.37222222222222223, 0.37222222222222223, 0.38333333333333336, 0.3388888888888889, 0.5166666666666667, 0.6444444444444445, 0.7611111111111111, 0.5222222222222223, 0.65, 0.7722222222222223, 0.32222222222222224, 0.3277777777777778, 0.3333333333333333]
    #     return x_right, y_right

    def set_config(self, id_event):
        """id_event is number of event what is series of golf swing
        Args:
            id_event (int): number of events

        """
        if id_event==0:
            x_right = [0.5833333333333334, 0.55, 0.5277777777777778, 0.4222222222222222, 0.45, 0.5777777777777777, 0.45,0.4777777777777778, 0.45, 0.5111111111111111, 0.46111111111111114, 0.4777777777777778, 0.49444444444444446, 0.4444444444444444, 0.5833333333333334, 0.5888888888888889, 0.5777777777777777, 0.5722222222222222]
            y_right = [0.38333333333333336, 0.43333333333333335, 0.4166666666666667, 0.4388888888888889, 0.37222222222222223, 0.4444444444444444, 0.43333333333333335, 0.38333333333333336, 0.6222222222222222, 0.75, 0.9, 0.6277777777777778, 0.75, 0.8555555555555555, 0.37777777777777777, 0.38333333333333336, 0.3888888888888889, 0.4]
        if id_event==1:
            x_right = [0.5833333333333334, 0.55, 0.5277777777777778, 0.4222222222222222, 0.45, 0.5777777777777777, 0.45,0.4777777777777778, 0.45, 0.5111111111111111, 0.46111111111111114, 0.4777777777777778, 0.49444444444444446, 0.4444444444444444, 0.5833333333333334, 0.5888888888888889, 0.5777777777777777, 0.5722222222222222]
            y_right = [0.38333333333333336, 0.43333333333333335, 0.4166666666666667, 0.4388888888888889, 0.37222222222222223, 0.4444444444444444, 0.43333333333333335, 0.38333333333333336, 0.6222222222222222, 0.75, 0.9, 0.6277777777777778, 0.75, 0.8555555555555555, 0.37777777777777777, 0.38333333333333336, 0.3888888888888889, 0.4]
        if id_event==2:
            x_right = [0.5833333333333334, 0.55, 0.5277777777777778, 0.4222222222222222, 0.45, 0.5777777777777777, 0.45,0.4777777777777778, 0.45, 0.5111111111111111, 0.46111111111111114, 0.4777777777777778, 0.49444444444444446, 0.4444444444444444, 0.5833333333333334, 0.5888888888888889, 0.5777777777777777, 0.5722222222222222]
            y_right = [0.38333333333333336, 0.43333333333333335, 0.4166666666666667, 0.4388888888888889, 0.37222222222222223, 0.4444444444444444, 0.43333333333333335, 0.38333333333333336, 0.6222222222222222, 0.75, 0.9, 0.6277777777777778, 0.75, 0.8555555555555555, 0.37777777777777777, 0.38333333333333336, 0.3888888888888889, 0.4]
        if id_event==3:
            x_right = [0.5833333333333334, 0.55, 0.5277777777777778, 0.4222222222222222, 0.45, 0.5777777777777777, 0.45,0.4777777777777778, 0.45, 0.5111111111111111, 0.46111111111111114, 0.4777777777777778, 0.49444444444444446, 0.4444444444444444, 0.5833333333333334, 0.5888888888888889, 0.5777777777777777, 0.5722222222222222]
            y_right = [0.38333333333333336, 0.43333333333333335, 0.4166666666666667, 0.4388888888888889, 0.37222222222222223, 0.4444444444444444, 0.43333333333333335, 0.38333333333333336, 0.6222222222222222, 0.75, 0.9, 0.6277777777777778, 0.75, 0.8555555555555555, 0.37777777777777777, 0.38333333333333336, 0.3888888888888889, 0.4]
        if id_event==4:
            x_right = [0.5833333333333334, 0.55, 0.5277777777777778, 0.4222222222222222, 0.45, 0.5777777777777777, 0.45,0.4777777777777778, 0.45, 0.5111111111111111, 0.46111111111111114, 0.4777777777777778, 0.49444444444444446, 0.4444444444444444, 0.5833333333333334, 0.5888888888888889, 0.5777777777777777, 0.5722222222222222]
            y_right = [0.38333333333333336, 0.43333333333333335, 0.4166666666666667, 0.4388888888888889, 0.37222222222222223, 0.4444444444444444, 0.43333333333333335, 0.38333333333333336, 0.6222222222222222, 0.75, 0.9, 0.6277777777777778, 0.75, 0.8555555555555555, 0.37777777777777777, 0.38333333333333336, 0.3888888888888889, 0.4]
        if id_event==5:
            x_right = [0.5833333333333334, 0.55, 0.5277777777777778, 0.4222222222222222, 0.45, 0.5777777777777777, 0.45,0.4777777777777778, 0.45, 0.5111111111111111, 0.46111111111111114, 0.4777777777777778, 0.49444444444444446, 0.4444444444444444, 0.5833333333333334, 0.5888888888888889, 0.5777777777777777, 0.5722222222222222]
            y_right = [0.38333333333333336, 0.43333333333333335, 0.4166666666666667, 0.4388888888888889, 0.37222222222222223, 0.4444444444444444, 0.43333333333333335, 0.38333333333333336, 0.6222222222222222, 0.75, 0.9, 0.6277777777777778, 0.75, 0.8555555555555555, 0.37777777777777777, 0.38333333333333336, 0.3888888888888889, 0.4]
        if id_event==6:
            x_right = [0.5833333333333334, 0.55, 0.5277777777777778, 0.4222222222222222, 0.45, 0.5777777777777777, 0.45,0.4777777777777778, 0.45, 0.5111111111111111, 0.46111111111111114, 0.4777777777777778, 0.49444444444444446, 0.4444444444444444, 0.5833333333333334, 0.5888888888888889, 0.5777777777777777, 0.5722222222222222]
            y_right = [0.38333333333333336, 0.43333333333333335, 0.4166666666666667, 0.4388888888888889, 0.37222222222222223, 0.4444444444444444, 0.43333333333333335, 0.38333333333333336, 0.6222222222222222, 0.75, 0.9, 0.6277777777777778, 0.75, 0.8555555555555555, 0.37777777777777777, 0.38333333333333336, 0.3888888888888889, 0.4]
        elif id_event==7:
            x_right = [0.5833333333333334, 0.55, 0.5277777777777778, 0.4222222222222222, 0.45, 0.5777777777777777, 0.45,0.4777777777777778, 0.45, 0.5111111111111111, 0.46111111111111114, 0.4777777777777778, 0.49444444444444446, 0.4444444444444444, 0.5833333333333334, 0.5888888888888889, 0.5777777777777777, 0.5722222222222222]
            y_right = [0.38333333333333336, 0.43333333333333335, 0.4166666666666667, 0.4388888888888889, 0.37222222222222223, 0.4444444444444444, 0.43333333333333335, 0.38333333333333336, 0.6222222222222222, 0.75, 0.9, 0.6277777777777778, 0.75, 0.8555555555555555, 0.37777777777777777, 0.38333333333333336, 0.3888888888888889, 0.4]
        return x_right, y_right

    def load_list(self, folder):
        images = []
        for filename in sorted(os.listdir(folder)):
            img = cv2.imread(os.path.join(folder, filename))
            if img is not None:
                print('path {}'.format(os.path.join(folder, filename)))
                images.append(img)
        return np.asarray(images)

    def score_infer(self, path):
        img_list = self.load_list(path)
        fig = plt.figure()
        for idx, img in enumerate(img_list):
            list_score = []
            print('image shape {}'.format(img.shape))
            img = Image.fromarray(img)
            print('image type {} and mode {}'.format(type(img), img.mode))
            # get vector from pose estimation
            img, x_vector, y_vector, number_keypoint= scr.score(img, idx)

            score = self.cosine_distane(x_vector, y_vector, idx, number_keypoint)
            list_score.append(score)
            a = fig.add_subplot(2, 4, idx + 1)
            a.set_title('score' + str(score))
            plt.imshow(img)
        plt.show()

    def cosine_distane(self, x_vector, y_vector, event, number_keypoint):
        """caculate score from vector_a, vector_b
        Args:
            x_vector (list): included x of point 
            y_vector (list): included y of point 
            event (int)): id of event 
        """
        
        event_score = []
        x_right_keypoint = []
        y_right_keypoint = []
        x_right, y_right = self.set_config(event)
        for id, value in enumerate(number_keypoint):
            print(value)
            x_right_keypoint.append(x_right[value])
            y_right_keypoint.append(y_right[value])
        x_score = 1 - dis.cosine(x_vector, x_right_keypoint)
        y_score = 1 - dis.cosine(y_vector, y_right_keypoint)
        event_score = (np.average([x_score, y_score]))
        return '%0.2f' % event_score








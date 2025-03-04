'''code adapted from https://github.com/experiencor/keras-yolo3'''
import numpy as np
import os
import cv2

class BoundBox:
    def __init__(self, xmin, ymin, xmax, ymax, c = None, classes = None):
        self.xmin = xmin 
        self.ymin = ymin 
        self.xmax = xmax
        self.ymax = ymax
        
        self.c       = c
        self.classes = classes

        self.label = -1
        self.score = -1

    def get_label(self):
        if self.label == -1:
            self.label = np.argmax(self.classes)
        
        return self.label
    
    def get_score(self):
        if self.score == -1:
            self.score = self.classes[self.get_label()]
            
        return self.score      

def _interval_overlap(interval_a, interval_b):
    
    x1, x2 = interval_a
    x3, x4 = interval_b
    # overlap is lesser of x4, x2 minus the greater of x3, x1
    overlap = min(x4,x2) - max(x3,x1)
    # overlap should be >= 0 (otherwise there is no overlap)
    return max(overlap, 0)    

def bbox_iou(box1, box2):
    
    intersect_w = _interval_overlap([box1.xmin, box1.xmax], [box2.xmin, box2.xmax])
    intersect_h = _interval_overlap([box1.ymin, box1.ymax], [box2.ymin, box2.ymax])  
    
    intersect = intersect_w * intersect_h
    
    w1, h1 = box1.xmax-box1.xmin, box1.ymax-box1.ymin
    w2, h2 = box2.xmax-box2.xmin, box2.ymax-box2.ymin
    
    union = w1*h1 + w2*h2 - intersect
    iou = float(intersect) / union if union > 0 else 0
    return iou
 

def draw_boxes_gt(image, boxes):
    ''' Draw ground truth boxes'''
    for box in boxes:
        xmin = box[1]
        xmax = box[3]
        ymin = box[2]
        ymax = box[4]

        cv2.rectangle(image, (xmin,ymin), (xmax,ymax), (0,0,255), 2)
        
    return image    
def draw_boxes(image, boxes, labels, obj_thresh, quiet=True):
    accepted_boxes =[]
    for box in boxes:
        label_str = ''
        label = -1
        
        for i in range(len(labels)):
            if box.classes[i] > obj_thresh:
                accepted_boxes.append(box)
                if label_str != '': label_str += ', '
                label_str += (labels[i] + ' ' + str(round(box.get_score(), 2)))
                label = i
            if not quiet: print(label_str)
                
        if label >= 0:
            text_size = cv2.getTextSize(label_str, cv2.FONT_HERSHEY_SIMPLEX, 1.1e-3 * image.shape[0], 5)
            width, height = text_size[0][0], text_size[0][1]
            region = np.array([[box.xmin-3,        box.ymin], 
                               [box.xmin-3,        box.ymin-height-26], 
                               [box.xmin+width+13, box.ymin-height-26], 
                               [box.xmin+width+13, box.ymin]], dtype='int32')  

            cv2.rectangle(img=image, pt1=(box.xmin,box.ymin), pt2=(box.xmax,box.ymax), color=(0,255,0), thickness=2)
            #cv2.fillPoly(img=image, pts=[region], color=get_color(label))
            cv2.putText(img=image, 
                        text=label_str, 
                        org=(box.xmin-3, box.ymin - 3), 
                        fontFace=cv2.FONT_HERSHEY_SIMPLEX, 
                        fontScale=7e-4 * image.shape[0], 
                        color=(0,255,0), 
                        thickness=2)
        
    return image ,accepted_boxes         
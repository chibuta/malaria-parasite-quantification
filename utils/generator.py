import cv2
import copy
import numpy as np
from keras.utils import Sequence
import imgaug as ia
from imgaug import augmenters as iaa
from utils.bbox import BoundBox, bbox_iou


class create_batch(Sequence):
    def __init__(self, 
        instances, 
        anchors,   
        labels,        
        downsample, # ratio between network input's size and network output's size, 32 for YOLOv3
        max_box_per_image,
        batch_size,
        min_net_size,
        max_net_size,
        net_size,
        shuffle, 
        jitter, 
        norm
    ):
        self.instances          = instances
        self.batch_size         = batch_size
        self.labels             = labels
        self.downsample         = downsample
        self.max_box_per_image  = max_box_per_image
        self.min_net_size       = (min_net_size//self.downsample)*self.downsample
        self.max_net_size       = (max_net_size//self.downsample)*self.downsample
        self.shuffle            = shuffle
        self.jitter             = jitter
        self.norm               = norm
        self.anchors            = [BoundBox(0, 0, anchors[2*i], anchors[2*i+1]) for i in range(len(anchors)//2)]
        self.net_h              = net_size  
        self.net_w              = net_size    

        self.aug                = True
        
        sometimes = lambda aug: iaa.Sometimes(0.5, aug)

        # Define our sequence of augmentation steps that will be applied to every image
        # All augmenters with per_channel=0.5 will sample one value _per image_
        # in 50% of all cases. In all other cases they will sample new values
        # _per channel_.
        
        self.aug_pipe = iaa.Sequential(
            [   
                # apply the following augmenters to most images
                iaa.Fliplr(0.5), # horizontally flip 50% of all images
                iaa.Flipud(0.5), # vertically flip 20% of all images
                # crop images by -5% to 10% of their height/width
                sometimes(iaa.CropAndPad(
                    percent=(-0.05, 0.1),
                    pad_mode=ia.ALL,
                    pad_cval=(0, 255)
                )),
                sometimes(iaa.Affine(
                    scale={"x": (0.8, 1.2), "y": (0.8, 1.2)}, # scale images to 80-120% of their size, individually per axis
                    translate_percent={"x": (-0.2, 0.2), "y": (-0.2, 0.2)}, # translate by -20 to +20 percent (per axis)
                    rotate=(-40, 40), # rotate by -45 to +45 degrees
                    shear=(-10, 10), # shear by -16 to +16 degrees
                    order=[0, 1], # use nearest neighbour or bilinear interpolation (fast)
                    cval=(0, 255), # if mode is constant, use a cval between 0 and 255
                    mode=ia.ALL # use any of scikit-image's warping modes (see 2nd image from the top for examples)
                )),
                # execute 0 to 5 of the following (less important) augmenters per image
                # don't execute all of them, as that would often be way too strong
                iaa.SomeOf((0, 5),
                    [
                        sometimes(iaa.Superpixels(p_replace=(0, 1.0), n_segments=(20, 200))), # convert images into their superpixel representation
                        iaa.OneOf([
                            iaa.GaussianBlur((0, 3.0)), # blur images with a sigma between 0 and 3.0
                            iaa.AverageBlur(k=(2, 7)), # blur image using local means with kernel sizes between 2 and 7
                            iaa.MedianBlur(k=(3, 11)), # blur image using local medians with kernel sizes between 2 and 7
                        ]),
                        iaa.Sharpen(alpha=(0, 1.0), lightness=(0.75, 1.5)), # sharpen images
                        iaa.Emboss(alpha=(0, 1.0), strength=(0, 2.0)), # emboss images
                        # search either for all edges or for directed edges,
                        # blend the result with the original image using a blobby mask
                        iaa.SimplexNoiseAlpha(iaa.OneOf([
                            iaa.EdgeDetect(alpha=(0.5, 1.0)),
                            iaa.DirectedEdgeDetect(alpha=(0.5, 1.0), direction=(0.0, 1.0)),
                        ])),
                        iaa.AdditiveGaussianNoise(loc=0, scale=(0.0, 0.05*255), per_channel=0.5), # add gaussian noise to images
                        iaa.OneOf([
                            iaa.Dropout((0.01, 0.1), per_channel=0.5), # randomly remove up to 10% of the pixels
                            iaa.CoarseDropout((0.03, 0.15), size_percent=(0.02, 0.05), per_channel=0.2),
                        ]),
                        iaa.Invert(0.05, per_channel=True), # invert color channels
                        iaa.Add((-10, 10), per_channel=0.5), # change brightness of images (by -10 to 10 of original value)
                        iaa.AddToHueAndSaturation((-20, 20)), # change hue and saturation
                        # either change the brightness of the whole image (sometimes
                        # per channel) or change the brightness of subareas
                        iaa.OneOf([
                            iaa.Multiply((0.5, 1.5), per_channel=0.5),
                            iaa.FrequencyNoiseAlpha(
                                exponent=(-4, 0),
                                first=iaa.Multiply((0.5, 1.5), per_channel=True),
                                second=iaa.ContrastNormalization((0.5, 2.0))
                            )
                        ]),
                        iaa.ContrastNormalization((0.5, 2.0), per_channel=0.5), # improve or worsen the contrast
                        iaa.Grayscale(alpha=(0.0, 1.0)),
                        #sometimes(iaa.ElasticTransformation(alpha=(0.5, 3.5), sigma=0.25)), # move pixels locally around (with random strengths)
                        #sometimes(iaa.PiecewiseAffine(scale=(0.01, 0.05))) # sometimes move parts of the image around
                    ],
                    random_order=True
                )
            ],
            random_order=True
        )
        if shuffle: np.random.shuffle(self.instances)
            
    def __len__(self):
        return int(np.ceil(float(len(self.instances))/self.batch_size))           

    def __getitem__(self, idx):
        # get image input size, change every 10 batches
        if self.min_net_size < self.max_net_size:
            net_h, net_w = self._get_net_size(idx) 
        else: 
            net_h, net_w = self.net_h, self.net_w
        
        self.aug = True
        
        base_grid_h, base_grid_w = net_h//self.downsample, net_w//self.downsample

        # determine the first and the last indices of the batch
        l_bound = idx*self.batch_size
        r_bound = (idx+1)*self.batch_size

        if r_bound > len(self.instances):
            r_bound = len(self.instances)
            l_bound = r_bound - self.batch_size

        x_batch = np.zeros((r_bound - l_bound, net_h, net_w, 3))             # input images
        t_batch = np.zeros((r_bound - l_bound, 1, 1, 1,  self.max_box_per_image, 4))   # list of groundtruth boxes

        # initialize the inputs and the outputs
        yolo_1 = np.zeros((r_bound - l_bound, 1*base_grid_h,  1*base_grid_w, len(self.anchors)//3, 4+1+len(self.labels))) # desired network output 1
        yolo_2 = np.zeros((r_bound - l_bound, 2*base_grid_h,  2*base_grid_w, len(self.anchors)//3, 4+1+len(self.labels))) # desired network output 2
        yolo_3 = np.zeros((r_bound - l_bound, 4*base_grid_h,  4*base_grid_w, len(self.anchors)//3, 4+1+len(self.labels))) # desired network output 3
        yolos = [yolo_3, yolo_2, yolo_1]

        dummy_yolo_1 = np.zeros((r_bound - l_bound, 1))
        dummy_yolo_2 = np.zeros((r_bound - l_bound, 1))
        dummy_yolo_3 = np.zeros((r_bound - l_bound, 1))
        
        instance_count = 0
        true_box_index = 0

        # do the logic to fill in the inputs and the output
        for train_instance in self.instances[l_bound:r_bound]:
            # augment input image and fix object's position and size
            img, all_objs = self._aug_image(train_instance, net_h, net_w,idx)
            
            for obj in all_objs:
                # find the best anchor box for this object
                try:
                    max_anchor = None                
                    max_index  = -1
                    max_iou    = -1

                    shifted_box = BoundBox(0, 
                                           0,
                                           obj['xmax']-obj['xmin'],                                                
                                           obj['ymax']-obj['ymin'])    

                    for i in range(len(self.anchors)):
                        anchor = self.anchors[i]
                        iou    = bbox_iou(shifted_box, anchor)

                        if max_iou < iou:
                            max_anchor = anchor
                            max_index  = i
                            max_iou    = iou                

                    # determine the yolo to be responsible for this bounding box
                    yolo = yolos[max_index//3]
                    grid_h, grid_w = yolo.shape[1:3]

                    # determine the position of the bounding box on the grid
                    center_x = .5*(obj['xmin'] + obj['xmax'])
                    center_x = center_x / float(net_w) * grid_w # sigma(t_x) + c_x
                    center_y = .5*(obj['ymin'] + obj['ymax'])
                    center_y = center_y / float(net_h) * grid_h # sigma(t_y) + c_y

                    # determine the sizes of the bounding box
                    wW =(obj['xmax'] - obj['xmin']) / float(max_anchor.xmax)
                    hH=(obj['ymax'] - obj['ymin']) / float(max_anchor.ymax)
                    w = np.log(wW) if wW>0 else 0 # t_w
                    h = np.log(hH) if hH>0 else 0# t_h

                    box = [center_x, center_y, w, h]

                    # determine the index of the label
                    obj_indx = self.labels.index(obj['name'])  

                    # determine the location of the cell responsible for this object
                    grid_x = int(np.floor(center_x))
                    grid_y = int(np.floor(center_y))

                    # assign ground truth x, y, w, h, confidence and class probs to y_batch
                    yolo[instance_count, grid_y, grid_x, max_index%3, 0:4] = box
                    yolo[instance_count, grid_y, grid_x, max_index%3, 4  ] = 1.
                    yolo[instance_count, grid_y, grid_x, max_index%3, 5+obj_indx] = 1

                    # assign the true box to t_batch
                    true_box = [center_x, center_y, obj['xmax'] - obj['xmin'], obj['ymax'] - obj['ymin']]
                    t_batch[instance_count, 0, 0, 0, true_box_index] = true_box

                    true_box_index += 1
                    true_box_index  = true_box_index % self.max_box_per_image  
                except:
                    continue

            # assign input image to x_batch
            if self.norm != None: 
                x_batch[instance_count] = self.norm(img)
            else:
                # plot image and bounding boxes for sanity check
                for obj in all_objs:
                    
                    cv2.rectangle(img[:,:,::-1], (obj['xmin'],obj['ymin']), (obj['xmax'],obj['ymax']), (255,0,0), 1)
                    cv2.putText(img[:,:,::-1], obj['name'], 
                                (obj['xmin']-3, obj['ymin']-3), 
                                0, 7e-4 * img.shape[0], 
                                (0,255,0), 1)
                
                x_batch[instance_count] = img

            # increase instance counter in the current batch
            instance_count += 1                 
                
        return [x_batch, t_batch, yolo_1, yolo_2, yolo_3], [dummy_yolo_1, dummy_yolo_2, dummy_yolo_3]

    def _get_net_size(self, idx):
        if idx%10 == 0:
            #self.aug = False
            net_size = self.downsample*np.random.randint(self.min_net_size/self.downsample, \
                                                         self.max_net_size/self.downsample+1)
            #print("resizing: ",idx, net_size, net_size)
            self.net_h, self.net_w = net_size, net_size
        return self.net_h, self.net_w
    
    
    def _aug_image(self, instance, net_h, net_w,idx):
        image_name = instance['filename']
        image = cv2.imread(image_name) # RGB image
       
        if image is None: print ('Cannot find ', image_name)

        h, w, c = image.shape
        all_objs = copy.deepcopy(instance['object'])
        
        rnd  = np.random.uniform()
        if self.jitter and rnd > 0.8:
 
            #Augment using imaug https://github.com/aleju/imgaug
            seq_det = self.aug_pipe.to_deterministic()
            image_aug = seq_det.augment_image(image)
            bboxes = ia.BoundingBoxesOnImage([
                                             ia.BoundingBox(x1=int(obj['xmin']), 
                                                            y1=int(obj['ymin']), 
                                                            x2=int(obj['xmax']), 
                                                            y2=int(obj['ymax'])) for obj in all_objs
                                              ], shape=image.shape)

            bboxes =  seq_det.augment_bounding_boxes([bboxes])[0] #fix bounding boxes after aug   
            image_aug = ia.imresize_single_image(image_aug, (net_h, net_w))
            bboxes = bboxes.on(image_aug)
            
            #Fix bounding boxes
            i = 0
            for obj in all_objs:
                obj['xmin'] =int(bboxes.bounding_boxes[i].x1)
                obj['ymin'] =int(bboxes.bounding_boxes[i].y1)
                obj['xmax'] =int(bboxes.bounding_boxes[i].x2)
                obj['ymax'] =int(bboxes.bounding_boxes[i].y2)
                i =i+1
                
            image = image_aug[:,:,::-1]

        else:
            
            
            seq_det = self.aug_pipe.to_deterministic()
            bboxes = ia.BoundingBoxesOnImage([
                                             ia.BoundingBox(x1=int(obj['xmin']), 
                                                            y1=int(obj['ymin']), 
                                                            x2=int(obj['xmax']), 
                                                            y2=int(obj['ymax'])) for obj in all_objs
                                              ], shape=image.shape)

             
            image_aug = ia.imresize_single_image(image, (net_h, net_w))
            bboxes = bboxes.on(image_aug)
            
            #Fix bounding boxes
            i = 0
            for obj in all_objs:
                obj['xmin'] =int(bboxes.bounding_boxes[i].x1)
                obj['ymin'] =int(bboxes.bounding_boxes[i].y1)
                obj['xmax'] =int(bboxes.bounding_boxes[i].x2)
                obj['ymax'] =int(bboxes.bounding_boxes[i].y2)
                i =i+1
                
            image = image_aug[:,:,::-1]
       
            
        
        return image, all_objs   

    def on_epoch_end(self):
        if self.shuffle: np.random.shuffle(self.instances)
            
    def num_classes(self):
        return len(self.labels)

    def size(self):
        return len(self.instances)    

    def get_anchors(self):
        anchors = []

        for anchor in self.anchors:
            anchors += [anchor.xmax, anchor.ymax]

        return anchors

    def load_annotation(self, i):
        annots = []

        for obj in self.instances[i]['object']:
            annot = [obj['xmin'], obj['ymin'], obj['xmax'], obj['ymax'], self.labels.index(obj['name'])]
            annots += [annot]

        if len(annots) == 0: annots = [[]]

        return np.array(annots)

    def load_image(self, i):
        return cv2.imread(self.instances[i]['filename'])     
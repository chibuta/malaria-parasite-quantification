'''Majority of the code here is adapted from https://github.com/experiencor/keras-yolo3'''

from keras.layers import Conv2D, Input, Activation,BatchNormalization,LeakyReLU, ZeroPadding2D, UpSampling2D, Lambda,SeparableConv2D,DepthwiseConv2D
from keras.layers.merge import add, concatenate
from keras.models import Model
from keras.utils.vis_utils import plot_model
from keras.engine.topology import Layer
from keras.regularizers import l2
from keras import backend as K
import tensorflow as tf


class yolo_layer(Layer):
    def __init__(self, anchors, max_grid, batch_size, warmup_batches, ignore_thresh, 
                    grid_scale, obj_scale, noobj_scale, xywh_scale, class_scale, 
                    **kwargs):
        # make the model settings persistent
        self.ignore_thresh  = ignore_thresh
        self.warmup_batches = warmup_batches
        self.anchors        = tf.constant(anchors, dtype='float', shape=[1,1,1,3,2])
        self.grid_scale     = grid_scale
        self.obj_scale      = obj_scale
        self.noobj_scale    = noobj_scale
        self.xywh_scale     = xywh_scale
        self.class_scale    = class_scale        

        # make a persistent mesh grid
        max_grid_h, max_grid_w = max_grid

        cell_x = tf.to_float(tf.reshape(tf.tile(tf.range(max_grid_w), [max_grid_h]), (1, max_grid_h, max_grid_w, 1, 1)))
        cell_y = tf.transpose(cell_x, (0,2,1,3,4))
        self.cell_grid = tf.tile(tf.concat([cell_x,cell_y],-1), [batch_size, 1, 1, 3, 1])

        super(yolo_layer, self).__init__(**kwargs)

    def build(self, input_shape):
        super(yolo_layer, self).build(input_shape)  # Be sure to call this somewhere!

    def call(self, x):
        input_image, y_pred, y_true, true_boxes = x

        # adjust the shape of the y_predict [batch, grid_h, grid_w, 3, 4+1+nb_class]
        y_pred = tf.reshape(y_pred, tf.concat([tf.shape(y_pred)[:3], tf.constant([3, -1])], axis=0))
        
        # initialize the masks
        object_mask     = tf.expand_dims(y_true[..., 4], 4)

        # the variable to keep track of number of batches processed
        batch_seen = tf.Variable(0.)        

        # compute grid factor and net factor
        grid_h      = tf.shape(y_true)[1]
        grid_w      = tf.shape(y_true)[2]
        grid_factor = tf.reshape(tf.cast([grid_w, grid_h], tf.float32), [1,1,1,1,2])

        net_h       = tf.shape(input_image)[1]
        net_w       = tf.shape(input_image)[2]            
        net_factor  = tf.reshape(tf.cast([net_w, net_h], tf.float32), [1,1,1,1,2])
        
        """
        Adjust prediction
        """
        pred_box_xy    = (self.cell_grid[:,:grid_h,:grid_w,:,:] + tf.sigmoid(y_pred[..., :2]))  # sigma(t_xy) + c_xy
        pred_box_wh    = y_pred[..., 2:4]                                                       # t_wh
        pred_box_conf  = tf.expand_dims(tf.sigmoid(y_pred[..., 4]), 4)                          # adjust confidence
        pred_box_class = y_pred[..., 5:]                                                        # adjust class probabilities      

        """
        Adjust ground truth
        """
        true_box_xy    = y_true[..., 0:2] # (sigma(t_xy) + c_xy)
        true_box_wh    = y_true[..., 2:4] # t_wh
        true_box_conf  = tf.expand_dims(y_true[..., 4], 4)
        true_box_class = tf.argmax(y_true[..., 5:], -1)         

        """
        Compare each predicted box to all true boxes
        """        
        # initially, drag all objectness of all boxes to 0
        conf_delta  = pred_box_conf - 0 

        # then, ignore the boxes which have good overlap with some true box
        true_xy = true_boxes[..., 0:2] / grid_factor
        true_wh = true_boxes[..., 2:4] / net_factor
        
        true_wh_half = true_wh / 2.
        true_mins    = true_xy - true_wh_half
        true_maxes   = true_xy + true_wh_half
        
        pred_xy = tf.expand_dims(pred_box_xy / grid_factor, 4)
        pred_wh = tf.expand_dims(tf.exp(pred_box_wh) * self.anchors / net_factor, 4)
        
        pred_wh_half = pred_wh / 2.
        pred_mins    = pred_xy - pred_wh_half
        pred_maxes   = pred_xy + pred_wh_half    

        intersect_mins  = tf.maximum(pred_mins,  true_mins)
        intersect_maxes = tf.minimum(pred_maxes, true_maxes)

        intersect_wh    = tf.maximum(intersect_maxes - intersect_mins, 0.)
        intersect_areas = intersect_wh[..., 0] * intersect_wh[..., 1]
        
        true_areas = true_wh[..., 0] * true_wh[..., 1]
        pred_areas = pred_wh[..., 0] * pred_wh[..., 1]

        union_areas = pred_areas + true_areas - intersect_areas
        iou_scores  = tf.truediv(intersect_areas, union_areas)

        best_ious   = tf.reduce_max(iou_scores, axis=4)        
        conf_delta *= tf.expand_dims(tf.to_float(best_ious < self.ignore_thresh), 4)

        """
        Compute some online statistics
        """            
        true_xy = true_box_xy / grid_factor
        true_wh = tf.exp(true_box_wh) * self.anchors / net_factor

        true_wh_half = true_wh / 2.
        true_mins    = true_xy - true_wh_half
        true_maxes   = true_xy + true_wh_half

        pred_xy = pred_box_xy / grid_factor
        pred_wh = tf.exp(pred_box_wh) * self.anchors / net_factor 
        
        pred_wh_half = pred_wh / 2.
        pred_mins    = pred_xy - pred_wh_half
        pred_maxes   = pred_xy + pred_wh_half      

        intersect_mins  = tf.maximum(pred_mins,  true_mins)
        intersect_maxes = tf.minimum(pred_maxes, true_maxes)
        intersect_wh    = tf.maximum(intersect_maxes - intersect_mins, 0.)
        intersect_areas = intersect_wh[..., 0] * intersect_wh[..., 1]
        
        true_areas = true_wh[..., 0] * true_wh[..., 1]
        pred_areas = pred_wh[..., 0] * pred_wh[..., 1]

        union_areas = pred_areas + true_areas - intersect_areas
        iou_scores  = tf.truediv(intersect_areas, union_areas)
        iou_scores  = object_mask * tf.expand_dims(iou_scores, 4)
        
        count       = tf.reduce_sum(object_mask)
        count_noobj = tf.reduce_sum(1 - object_mask)
        detect_mask = tf.to_float((pred_box_conf*object_mask) >= 0.5)
        class_mask  = tf.expand_dims(tf.to_float(tf.equal(tf.argmax(pred_box_class, -1), true_box_class)), 4)
        recall50    = tf.reduce_sum(tf.to_float(iou_scores >= 0.5 ) * detect_mask  * class_mask) / (count + 1e-3)
        recall75    = tf.reduce_sum(tf.to_float(iou_scores >= 0.75) * detect_mask  * class_mask) / (count + 1e-3)    
        avg_iou     = tf.reduce_sum(iou_scores) / (count + 1e-3)
        avg_obj     = tf.reduce_sum(pred_box_conf  * object_mask)  / (count + 1e-3)
        avg_noobj   = tf.reduce_sum(pred_box_conf  * (1-object_mask))  / (count_noobj + 1e-3)
        avg_cat     = tf.reduce_sum(object_mask * class_mask) / (count + 1e-3) 

        """
        Warm-up training
        """
        batch_seen = tf.assign_add(batch_seen, 1.)
        
        true_box_xy, true_box_wh, xywh_mask = tf.cond(tf.less(batch_seen, self.warmup_batches+1), 
                              lambda: [true_box_xy + (0.5 + self.cell_grid[:,:grid_h,:grid_w,:,:]) * (1-object_mask), 
                                       true_box_wh + tf.zeros_like(true_box_wh) * (1-object_mask), 
                                       tf.ones_like(object_mask)],
                              lambda: [true_box_xy, 
                                       true_box_wh,
                                       object_mask])

        """
        Compare each true box to all anchor boxes
        """      
        wh_scale = tf.exp(true_box_wh) * self.anchors / net_factor
        wh_scale = tf.expand_dims(2 - wh_scale[..., 0] * wh_scale[..., 1], axis=4) # the smaller the box, the bigger the scale

        xy_delta    = xywh_mask   * (pred_box_xy-true_box_xy) * wh_scale * self.xywh_scale
        wh_delta    = xywh_mask   * (pred_box_wh-true_box_wh) * wh_scale * self.xywh_scale
        conf_delta  = object_mask * (pred_box_conf-true_box_conf) * self.obj_scale + (1-object_mask) * conf_delta * self.noobj_scale
        class_delta = object_mask * \
                      tf.expand_dims(tf.nn.sparse_softmax_cross_entropy_with_logits(labels=true_box_class, logits=pred_box_class), 4) * \
                      self.class_scale

        loss_xy    = tf.reduce_sum(tf.square(xy_delta),       list(range(1,5)))
        loss_wh    = tf.reduce_sum(tf.square(wh_delta),       list(range(1,5)))
        loss_conf  = tf.reduce_sum(tf.square(conf_delta),     list(range(1,5)))
        loss_class = tf.reduce_sum(class_delta,               list(range(1,5)))

        loss = loss_xy + loss_wh + loss_conf + loss_class

        loss = tf.Print(loss, [grid_h, avg_obj], message='avg_obj \t\t', summarize=1000)
        loss = tf.Print(loss, [grid_h, avg_noobj], message='avg_noobj \t\t', summarize=1000)
        loss = tf.Print(loss, [grid_h, avg_iou], message='avg_iou \t\t', summarize=1000)
        loss = tf.Print(loss, [grid_h, avg_cat], message='avg_cat \t\t', summarize=1000)
        loss = tf.Print(loss, [grid_h, recall50], message='recall50 \t', summarize=1000)
        loss = tf.Print(loss, [grid_h, recall75], message='recall75 \t', summarize=1000)   
        loss = tf.Print(loss, [grid_h, count], message='count \t', summarize=1000)     
        loss = tf.Print(loss, [grid_h, tf.reduce_sum(loss_xy), 
                                       tf.reduce_sum(loss_wh), 
                                       tf.reduce_sum(loss_conf), 
                                       tf.reduce_sum(loss_class)],  message='loss xy, wh, conf, class: \t',   summarize=1000)   


        return loss*self.grid_scale

    def compute_output_shape(self, input_shape):
        return [(None, 1)]


def conv_block(inp, convs,weight_decay=1e-4):
    '''We adopt part of the mobilenetv2 architecture'''  
    x = inp

    for conv in convs:
        channel_axis = 1 if K.image_data_format() == 'filters_first' else -1 
        if conv['stride'] ==2 and conv['conv_id'] > 0:
            
            
            in_filters = K.int_shape(x)[channel_axis]
            x = Conv2D(in_filters  , 1, padding= 'same', 
                       strides=1, use_bias=False if conv['bnorm'] else True,
                       kernel_regularizer=l2(weight_decay), 
                       name='conv_' + str(conv['conv_id'])+'_0')(x)
            x = BatchNormalization(epsilon=1e-5, momentum=0.9, 
                                   name='conv_bn_'+str(conv['conv_id'])+'_0')(x)
            x = LeakyReLU(alpha=0.1, name='conv_leaky_' + str(conv['conv_id'])+'_0')(x)

            x = DepthwiseConv2D((3, 3),
                                padding='same',
                                depth_multiplier=2,
                                strides=conv['stride'],
                                use_bias=False if conv['bnorm'] else True,
                                kernel_regularizer=l2(weight_decay),
                                name='conv_dw_' + str(conv['conv_id'])+'_0' )(x)
            x = BatchNormalization(axis=channel_axis, epsilon=1e-5, 
                                   momentum=0.9, name='conv_dw_bn_'+str(conv['conv_id'])+'_0')(x)
            x = LeakyReLU(alpha=0.1, name='conv_dw_leaky_' + str(conv['conv_id'])+'_0')(x)
            x = Conv2D(conv['filter'], 1, padding='valid' if conv['stride'] > 1 else 'same', 
                       strides=1, use_bias=False if conv['bnorm'] else True,
                       kernel_regularizer=l2(weight_decay), 
                       name='conv_' + str(conv['conv_id'])+'_1')(x)
            x = BatchNormalization(axis=channel_axis, epsilon=1e-5, 
                                   momentum=0.9, name='conv_bn_'+str(conv['conv_id'])+'_1')(x)
            
            #bottleneck  
            x1 = LeakyReLU(alpha=0.1, name='conv_leaky_' + str(conv['conv_id'])+'_2')(x)
            x1 = DepthwiseConv2D((3, 3),
                                padding= 'same',
                                depth_multiplier=1,
                                strides=1,
                                use_bias=False if conv['bnorm'] else True,
                                kernel_regularizer=l2(weight_decay),
                                name='conv_dw_' + str(conv['conv_id'])+'_1')(x1)
            x1 = BatchNormalization(axis=channel_axis, epsilon=1e-5,momentum=0.9,
                                    name='conv_dw_bn_'+str(conv['conv_id'])+'_1')(x1)
            x1 = LeakyReLU(alpha=0.1, name='conv_dw_leaky_' + str(conv['conv_id'])+'_1')(x1)

            x1 = Conv2D(conv['filter'], 1, padding ='same', 
                        strides=1, use_bias=False,
                        kernel_regularizer=l2(weight_decay),name='conv_' + str(conv['conv_id'])+'_3')(x1)
            x1 = BatchNormalization(axis=channel_axis, epsilon=1e-5,
                                    momentum=0.9,name='conv_bn_'+str(conv['conv_id'])+'_3')(x1)

            x = add([x, x1], name='block_output'+str(conv['conv_id']))
            
        
        else:
            if conv['kernel'] ==3 and conv['stride'] ==1 and conv['conv_id'] >0:
                x = SeparableConv2D(conv['filter'],(3, 3),
                            padding='same',
                            depth_multiplier=1,
                            strides=1,
                            use_bias=False if conv['bnorm'] else True,
                            name='conv_dw_' + str(conv['conv_id']))(x)
            
            else:
                x = Conv2D(conv['filter'], 
                           conv['kernel'], 
                           strides=conv['stride'], 
                           padding= 'same', 
                           name='conv_' + str(conv['conv_id']), 
                       use_bias=False if conv['bnorm'] else True)(x)
            if conv['bnorm']: x = BatchNormalization(axis=channel_axis,epsilon=1e-5,
                                                     momentum=0.9, name='bnorm_' + str(conv['conv_id']))(x)
            if conv['leaky']: x = LeakyReLU(alpha=0.1, name='leaky_' + str(conv['conv_id']))(x)

    return   x      

def model(nb_class, 
          anchors, 
          max_grid,
          max_box_per_image = 30,  
          batch_size = 8, 
          warmup_batches = 0,
          ignore_thresh  = .3,
          grid_scales    = [4,2,1],
          obj_scale      = 5,
          noobj_scale    = 1,
          xywh_scale     = 1,
          class_scale    = 1):
    
    input_image = Input(shape=(None, None, 3)) # net_h, net_w, 3
    true_boxes  = Input(shape=(1, 1, 1, max_box_per_image, 4))
    true_yolo_1 = Input(shape=(None, None, len(anchors)//6, 4+1+nb_class)) # grid_h, grid_w, nb_anchor, 5+nb_class
    true_yolo_2 = Input(shape=(None, None, len(anchors)//6, 4+1+nb_class)) # grid_h, grid_w, nb_anchor, 5+nb_class
    true_yolo_3 = Input(shape=(None, None, len(anchors)//6, 4+1+nb_class)) # grid_h, grid_w, nb_anchor, 5+nb_class
    
    filters = [24,32,64,96,160,3*(5+nb_class)]
    
    
    x = conv_block(input_image,
                   [{'filter': filters[0], 'kernel': 3, 'stride': 2, 'bnorm': True, 'leaky': True, 'conv_id': 0},
                    {'filter': filters[1], 'kernel': 3, 'stride': 2, 'bnorm': True, 'leaky': True, 'conv_id': 1},
                    {'filter': filters[2], 'kernel': 3, 'stride': 2, 'bnorm': True, 'leaky': True, 'conv_id': 2}])
        
    skip_connection_1 = x
        
    
    x = conv_block(x, 
                   [{'filter': filters[3], 'kernel': 3, 'stride': 2, 'bnorm': True, 'leaky': True, 'conv_id': 3}])

        
    skip_connection_2 = x
        
    
    x = conv_block(x, 
                   [{'filter': filters[4], 'kernel': 3, 'stride': 2, 'bnorm': True, 'leaky': True, 'conv_id': 4}])
    
    

    
    pred_yolo_1 = conv_block(x, 
                          [{'filter': filters[4], 'kernel': 3, 'stride': 1, 'bnorm': True,  'leaky': True, 'conv_id': 5},
                           {'filter': filters[5], 'kernel': 1, 'stride': 1, 'bnorm': False, 'leaky': False, 'conv_id': 6}])
    
    loss_yolo_1 = yolo_layer(anchors[12:], 
                            [1*num for num in max_grid], 
                            batch_size, 
                            warmup_batches, 
                            ignore_thresh, 
                            grid_scales[0],
                            obj_scale,
                            noobj_scale,
                            xywh_scale,
                            class_scale)([input_image, pred_yolo_1, true_yolo_1, true_boxes])

    
    x = conv_block(x, [{'filter': filters[2], 'kernel': 1, 'stride': 1, 'bnorm': True, 'leaky': True, 'conv_id': 7}])
    x = UpSampling2D(2)(x)
    x = concatenate([x, skip_connection_2])

    
    x = conv_block(x, [{'filter': filters[2], 'kernel': 1, 'stride': 1, 'bnorm': True, 'leaky': True, 'conv_id': 8},
                        {'filter': filters[3], 'kernel': 3, 'stride': 1, 'bnorm': True, 'leaky': True, 'conv_id': 9},
                        {'filter': filters[2], 'kernel': 1, 'stride': 1, 'bnorm': True, 'leaky': True, 'conv_id': 10}])

    
    pred_yolo_2 = conv_block(x, 
                     [{'filter': filters[3], 'kernel': 3, 'stride': 1, 'bnorm': True,  'leaky': True,  'conv_id': 11},
                      {'filter': filters[5], 'kernel': 1, 'stride': 1, 'bnorm': False, 'leaky': False, 'conv_id': 12}])
    loss_yolo_2 = yolo_layer(anchors[6:12], 
                            [2*num for num in max_grid], 
                            batch_size, 
                            warmup_batches, 
                            ignore_thresh, 
                            grid_scales[1],
                            obj_scale,
                            noobj_scale,
                            xywh_scale,
                            class_scale)([input_image, pred_yolo_2, true_yolo_2, true_boxes])

   
    x = conv_block(x, [{'filter': filters[1], 'kernel': 1, 'stride': 1, 'bnorm': True, 'leaky': True,   'conv_id': 13}])
    x = UpSampling2D(2)(x)
    x = concatenate([x, skip_connection_1])

   
    pred_yolo_3 = conv_block(x, 
                        [{'filter': filters[1], 'kernel': 1, 'stride': 1, 'bnorm': True,  'leaky': True,  'conv_id': 14},
                         {'filter': filters[2], 'kernel': 3, 'stride': 1, 'bnorm': True,  'leaky': True,  'conv_id': 15},
                         {'filter': filters[1], 'kernel': 1, 'stride': 1, 'bnorm': True,  'leaky': True,  'conv_id': 16},
                         {'filter': filters[2], 'kernel': 3, 'stride': 1, 'bnorm': True,  'leaky': True,  'conv_id': 17},
                         {'filter': filters[5], 'kernel': 1, 'stride': 1, 'bnorm': False, 'leaky': False, 'conv_id': 18}])
    loss_yolo_3 = yolo_layer(anchors[:6], 
                            [4*num for num in max_grid], 
                            batch_size, 
                            warmup_batches, 
                            ignore_thresh, 
                            grid_scales[2],
                            obj_scale,
                            noobj_scale,
                            xywh_scale,
                            class_scale)([input_image, pred_yolo_3, true_yolo_3, true_boxes]) 

    train_model = Model([input_image, true_boxes, true_yolo_1, true_yolo_2, true_yolo_3], [loss_yolo_1, loss_yolo_2, loss_yolo_3])
    infer_model = Model(input_image, [pred_yolo_1, pred_yolo_2, pred_yolo_3])





    #train_model.summary()
    plot_model(train_model, to_file='training_model.png', show_shapes=True)
    plot_model(infer_model, to_file='inference_model.png', show_shapes=True)
    return [train_model, infer_model]

def dummy_loss(y_true, y_pred):
    return tf.sqrt(tf.reduce_sum(y_pred))
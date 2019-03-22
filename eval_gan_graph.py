"""Evaluation script for the SpinePathNet network on the validation subset
   of spine dataset.
This script evaluates the model on ? validation images.
"""

from __future__ import absolute_import
from __future__ import division
from __future__ import print_function
import scipy.misc
from datetime import datetime
import math
import os
from nets import preprocessing, SpinePathNet, metrics 
import tf_utils
import matplotlib.pyplot as plt
import numpy as np
import tensorflow as tf
from skimage.segmentation import slic
FLAGS = tf.app.flags.FLAGS

# Fold is one of five folds cross-validation. 
Fold = 1
os.environ["CUDA_VISIBLE_DEVICES"] = "2"
# =========================================================================== #
# Evaluation Flags.
# =========================================================================== #

tf.app.flags.DEFINE_string(
    'Fold', '1', 'Flod of 5-fold cross validation.')

tf.app.flags.DEFINE_string(
    'eval_dir', 'tmp/graph/tfmodels_gan_%s/'%FLAGS.Fold, 'Directory where the results are saved to.')

tf.app.flags.DEFINE_string('checkpoint_dir', 'tmp/graph/tfmodels_gan_%s/'%FLAGS.Fold,
                           """Directory where to read model checkpoints.""")

tf.app.flags.DEFINE_integer('eval_interval_secs', 60 * 5,
                            """How often to run the eval.""")

tf.app.flags.DEFINE_boolean('run_once', False,
                         """Whether to run eval only once.""")

# =========================================================================== #
# Dataset Flags.
# =========================================================================== #

tf.app.flags.DEFINE_string(
    'dataset_name', 'spine_segmentation', 'The name of the dataset to load.')
tf.app.flags.DEFINE_integer(
    'num_classes', 7, 'Number of classes to use in the dataset.')
tf.app.flags.DEFINE_string(
    'dataset_split_name', 'test_%s_fold'%FLAGS.Fold, 'The name of the train/test split and five folds.')
tf.app.flags.DEFINE_string(
    'dataset_dir', './datasets/tfrecords_spine_segmentation/', 'The directory where the dataset files are stored.')
tf.app.flags.DEFINE_integer(
    'batch_size', 1, 'The number of samples in each batch.')
tf.app.flags.DEFINE_integer(
    'num_samples', 50, 'The number of samples in the testing set.')
tf.app.flags.DEFINE_integer(
    'num_readers', 4,
    'The number of parallel readers that read data from the dataset.')
tf.app.flags.DEFINE_integer(
    'num_preprocessing_threads', 8,
    'The number of threads used to create the batches.')



def main(_):
    
    if not FLAGS.dataset_dir:
        
            raise ValueError('You must supply the dataset directory with --dataset_dir')
            
    with tf.Graph().as_default():
        
        # Select the dataset.
        dataset = FLAGS.dataset_dir + '%s_%s.tfrecord' %(FLAGS.dataset_name,FLAGS.dataset_split_name) 
        
        with tf.name_scope('input'):
            
            filename_queue = tf.train.string_input_producer([dataset], num_epochs= 1)
            
            image, mask_class, mask_instance = tf_utils.read_and_decode(filename_queue, batch_size = FLAGS.batch_size,\
                                                                       capacity=20 * FLAGS.batch_size,\
                                                                       num_threads=FLAGS.num_readers,\
                                                                       min_after_dequeue=10 * FLAGS.batch_size, is_training=False)
            #image = preprocessing.data_augmentation(image, is_training=False)
        # Generate feature_embedding for check and logits.
        feature_embedding, logits= SpinePathNet.g_net_graph(image, Fold, batch_size=FLAGS.batch_size,\
                                       class_num = FLAGS.num_classes, reuse=False,\
                                       is_training=False, scope='g_SpinePathNet')  

        
        
        ## The comparison between logits and ground truth. F1-score, IOU, pixel-acc, etc.
        #tf.nn.softmax(logits)
        pred = tf.argmax(tf.nn.softmax(logits), dimension = 3)
        
        
        #pred = tf.expand_dims(predication,dim=3)
        gt = mask_class
        
        #pred = tf.reshape(predication, [-1,])#
        #gt = tf.reshape(mask_class, [-1,])
        
        #weights = tf.cast(tf.greater_equal(gt, FLAGS.num_classes-1), tf.int32) #Ignoring sv class.
        weights = None
        # mIoU
        mIoU, update_IoU = tf.contrib.metrics.streaming_mean_iou(pred, gt, num_classes=FLAGS.num_classes, weights=weights)
        
        #Per class accu.
        PCA, update_PCA = tf.metrics.mean_per_class_accuracy(pred, gt, num_classes=FLAGS.num_classes, weights=weights)
        
        #Pixel-wise accuarcy.
        PWA, update_PWA = tf.metrics.accuracy(pred, gt, weights=weights)
        
        #streaming mean Dice score.
        mdc, mss, update_mds  =  metrics.mean_Dice_score(pred, gt, num_classes=FLAGS.num_classes)
        
        # The op for initializing the variables.                                                            
        init_op = tf.group(tf.global_variables_initializer(),
                           tf.local_variables_initializer())
        
        # Create a session for running operations in the Graph.
        #config = tf.ConfigProto(allow_soft_placement = True)
        sess = tf.Session()
        
        # Initialize the variables (the trained variables and the
        # epoch counter).
        sess.run(init_op)
        
        
        ###load weights.
        loader = tf.train.Saver()

        # Start input enqueue threads.
        coord = tf.train.Coordinator()
        threads = tf.train.start_queue_runners(sess=sess, coord=coord)
        
        ckpt = tf.train.get_checkpoint_state(FLAGS.checkpoint_dir)
        
        if ckpt and ckpt.model_checkpoint_path:
            # Restores from checkpoint
            loader.restore(sess, ckpt.model_checkpoint_path)
            #global_step = ckpt.model_checkpoint_path.split('/')[-1].split('-')[-1]
            print("Restored model parameters from {}".format(ckpt))
        else: 
            print('No checkpoint file found')
            
        num_iter = int(math.ceil(FLAGS.num_samples / FLAGS.batch_size))

        # Iterate over training steps.
        for step in xrange(num_iter):
            features,images,preds,gts,mdcs, msss, _,_,_,_ = sess.run([feature_embedding,image,pred,gt,mdc, mss, update_IoU, update_PCA, update_PWA, update_mds])

            if step % 40 == 0:
                print('step {:d}'.format(step))
                #predictions= np.array(preds)
                #predictions= np.reshape(predictions,(512,512))
                #plt.imsave('%spredictions_%d.jpg'%(FLAGS.eval_dir,step),predictions) 
                #gtss = np.array(gts)
                #gtss = np.reshape(gtss,(512,512))
                #plt.imsave(image_name,image,cmap='gray') 
                #plt.imsave('%sgt_%d.jpg'%(FLAGS.eval_dir,step),gtss) 
                images = np.array(images)
                images = np.reshape(images,(512,512))
                plt.imsave('%simage_%d.jpg'%(FLAGS.eval_dir,step),images)
                for i in xrange(128):
                    feature = np.array(features[:,:,:,i])
                    feature = np.reshape(feature,(256,256))
                    plt.imsave('%sfeature_al_%d_%d.jpg'%(FLAGS.eval_dir,step,i),feature)
                #plt.imsave('%sfeature_%d.jpg'%(FLAGS.eval_dir,step),feature,cmap='gray')
                #print(predictions)
                #np.savetxt('gtss.txt',gtss,fmt='%5s')
                #np.savetxt('preds.txt',predictions,fmt='%5s')                
        print('Mean IoU: {:.3f}'.format(mIoU.eval(session=sess)))#Session for mIOU.
        print('Mean Per Class Accuracy: {:.3f}'.format(PCA.eval(session=sess)))#Session for PCA.
        print('Mean Pixel-wise Accuracy: {:.9f}'.format(PWA.eval(session=sess)))#Session for PWA.
        print('Dice Score: {:s}'.format(mdcs))
        

        ## compute mean dice score without background.
        print('Mean Dice Score: {:.3f}'.format(reduce(lambda x, y: x + y, mdcs) / len(mdcs)))
        
        print('SS: {:s}'.format(msss))
        print('Mean Specificity Score: {:.3f}'.format((msss[0]+msss[2]+msss[4]) / 3.0))
        print('Mean Sensitivity Score: {:.3f}'.format((msss[1]+msss[3]+msss[5]) / 3.0))        
        
        coord.request_stop()
        coord.join(threads)
        sess.close()                          
       
    
    
if __name__ == '__main__':

    tf.app.run()        
        
        
        
        
        

        
        
        
        
        
        
        
        
        
        
        
        
        
        
        
        
        


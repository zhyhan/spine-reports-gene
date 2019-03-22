from __future__ import absolute_import
from __future__ import division
from __future__ import print_function
import os
import numpy as np
import tqdm
import sys
from nets import preprocessing, SpinePathNet, metrics
from datasets import convert_dicom_to_tfrecord as cdtt
# from mpl_toolkits.axes_grid1 import make_axes_locatable #colorbar whose height (or width) in sync with the master axe
from matplotlib.colors import LinearSegmentedColormap
import matplotlib.pyplot as plt
import tensorflow as tf
from skimage.measure import label
from skimage.measure import regionprops
sys.path.append('../')

class spinal_radiological_reports_generator():
    """
    It is to generate radiological reports for lumbar spines in MRIs.
    Input: predicted maps.
    Return: structured lists.    
    Three steps for post processing predicted maps:    
    Step 1: smooth pixels.     
    Step 2: labeling structures.    
    Step 3: generate structured lists, such as (L5, vertebra, normal).  
    """

    def __init__(self, prediction, class_number):
        # Below parameters are private by this obeject so called self.
        assert len(prediction.shape) == 2  # guarantee the shape of predictions
        assert np.max(prediction) == class_number - 1
        self.prediction = prediction
        self.weight = prediction.shape[1]
        self.height = prediction.shape[0]
        self.class_number = class_number
        self.SPINE_LABELS = {0: 'Background',
                             1: 'Normal Vertebral',
                             2: 'Vertebral Deformity',
                             3: 'Normal Disc',
                             4: 'Gegenerative Disc',
                             5: 'Normal Neural Foraminal',
                             6: 'Neural Foraminal Stenosis'}

    def summarize_pixel_one_structure(self, spinal_structure):
        """
        Input: spinal structure.
        Return: segmented_points.
        """
        labels = []
        mapfile = self.prediction
        map = mapfile.copy()
        if spinal_structure is "vertebrae":
            reserve = 2
            area_threshold = 1000
        elif spinal_structure is "disc":
            reserve = 4
            area_threshold = 200
        elif spinal_structure is "foramen":
            reserve = 6
            area_threshold = 60
        else:
            print("No spinal structure in this task called: %s" % spinal_structure)
        map[map == reserve - 1] = reserve
        map[map != reserve] = 0
        label_map = label(map)
        props = regionprops(label_map)
        spinal_order = []
        for z, j in enumerate(props):
            if j.area >= area_threshold:
                spinal_order.append(z)
        if len(spinal_order) > 5:
            spinal_order = spinal_order[-5:]  # reserve the final five structures for the report generation.
        elif len(spinal_order) < 5:
            raise ValueError("The threshold is unsuitable!")
        else:
            spinal_order = spinal_order
        for order in spinal_order:
            normal_amount = 0
            abnormal_amount = 0
            coords = props[order].coords
            for coord in coords:
                if mapfile[coord[0], coord[1]] == reserve:
                    abnormal_amount += 1
                elif mapfile[coord[0], coord[1]] == reserve - 1:
                    normal_amount += 1
            if abnormal_amount >= normal_amount:
                labels.append(1)
            else:
                labels.append(0)
        return labels

    def summarize_pixel_all_structure(self):
        """
        Input None,
        Return three types of spinal structure.
        """
        disc_condition = self.summarize_pixel_one_structure("disc")
        foramen_condition = self.summarize_pixel_one_structure("foramen")
        vertebrae_condition = self.summarize_pixel_one_structure("vertebrae")
        return disc_condition, foramen_condition, vertebrae_condition

    def report_generation(self, filename):

        """
        Input: generate structured lists, such as (L5, vertebra, normal).
        Return: radiological reports.
        """
        radiological_report = []
        disc_condition, foramen_condition, vertebrae_condition = self.summarize_pixel_all_structure()
        num = len(disc_condition)
        if num == 5:
            pre_descriptions = "Automated generated radiological report for patient #{0}.".format(filename)
            radiological_report.append(pre_descriptions)
            MR_descriptions = "This spine is normally imaged in the sigattal direction."
            radiological_report.append(MR_descriptions)
            for i, j in enumerate(disc_condition):

                if j == 0 and foramen_condition[i] == 0 and vertebrae_condition[i] == 0:
                    if i < num - 1:
                        descriptions = "At L{0}-L{1}, the intervertebral disc does not have obvious degenerative changes. The neural foramina does noes have obvious stenosis. The above vertebra does not have deformative changes.".format(i+1, i+2)
                    else:
                        descriptions = "At L{0}-S1, the intervertebral disc does not have obvious degenerative changes. The neural foramina does noes have obvious stenosis. The above vertebra does not have deformative changes.".format(i+1)

                elif j == 0 and foramen_condition[i] == 0 and vertebrae_condition[i] == 1:
                    if i < num - 1:
                        descriptions = "At L{0}-L{1}, the intervertebral disc does not have obvious degenerative changes. The neural foramina does noes have obvious stenosis. The above vertebra has Deformative changes.".format(i+1, i+2)
                    else:
                        descriptions = "At L{0}-S1, the intervertebral disc does not have obvious degenerative changes. The neural foramina does noes have obvious stenosis. The above vertebra has Deformative changes.".format(i+1)

                elif j == 0 and foramen_condition[i] == 1 and vertebrae_condition[i] == 0:
                    if i < num - 1:
                        descriptions = "At L{0}-L{1}, the intervertebral disc does not have obvious degenerative changes. The neural foramina has obvious stenosis. The above vertebra does not have deformative changes.".format(i+1, i+2)
                    else:
                        descriptions = "At L{0}-S1, the intervertebral disc does not have obvious degenerative changes. The neural foramina has obvious stenosis. The above vertebra does not have deformative changes.".format(i+1)


                elif j == 1 and foramen_condition[i] == 0 and vertebrae_condition[i] == 0:
                    if i < num - 1:
                        descriptions = "At L{0}-L{1}, the intervertebral disc has obvious degenerative changes. The neural foramina does noes has obvious stenosis. The above vertebra does not have deformative changes.".format(i+1, i+2)
                    else:
                        descriptions = "At L{0}-S1, the intervertebral disc does has obvious degenerative changes. The neural foramina does noes has obvious stenosis. The above vertebra does not have deformative changes.".format(i+1)

                elif j == 1 and foramen_condition[i] == 1 and vertebrae_condition[i] == 1:
                    if i < num - 1:
                        descriptions = "At L{0}-L{1}, the intervertebral disc has obvious degenerative changes. The above vertebra also has deformative changes, which lead to neural foraminal stenosis to a certain extent.".format(i+1, i+2)
                    else:
                        descriptions = "At L{0}-S1, the intervertebral disc has obvious degenerative changes. The above vertebra also has deformative changes, which lead to neural foraminal stenosis to a certain extent.".format(i+1)

                elif j == 1 and foramen_condition[i] == 1 and vertebrae_condition[i] == 0:
                    if i < num - 1:
                        descriptions = "At L{0}-L{1}, disc degenerative changes are associated with neural foraminal stenosis. The above vertebra does not have deformative changes.".format(i+1, i+2)
                    else:
                        descriptions = "At L{0}-S1, disc degenerative changes are associated with neural foraminal stenosis. The above vertebra does not have deformative changes.".format(i+1)

                elif j == 0 and foramen_condition[i] == 1 and vertebrae_condition[i] == 1:
                    if i < num - 1:
                        descriptions = "At L{0}-L{1}, vertebra degenerative changes are associated with neural foraminal stenosis. The intervertebral disc does not have degenerative changes.".format(i+1, i+2)
                    else:
                        descriptions = "At L{0}-S1, disc degenerative changes are associated with neural foraminal stenosis. The intervertebral disc does not have degenerative changes.".format(i+1)

                elif j == 1 and foramen_condition[i] == 0 and vertebrae_condition[i] == 1:
                    if i < num - 1:
                        descriptions = "At L{0}-L{1}, the intervertebral disc has obvious degenerative changes. The neural foramina does noes has obvious stenosis. The above vertebra has deformative changes.".format(i+1, i+2)
                    else:
                        descriptions = "At L{0}-S1, the intervertebral disc does has obvious degenerative changes. The neural foramina does noes has obvious stenosis. The above vertebra has deformative changes.".format(i+1)
                radiological_report.append(descriptions)
        elif num < 5:
            MR_descriptions = "This spine is abnormally imaged in the sagittal direction."
            radiological_report.append(MR_descriptions)
        else:
            raise ValueError(
                'A very specific bad thing happened that the number of one type of diagnosed structure is larger than five!')

        return radiological_report



def discrete_cmap(N, base_cmap=None):
    """
    Create an N-bin discrete colormap
    from the specified input map

    """
    # Note that if base_cmap is a string or None, you can simply do
    #    return plt.cm.get_cmap(base_cmap, N)
    # The following works for string, None, or a colormap instance:

    base = plt.cm.get_cmap(base_cmap)
    color_list = base(np.linspace(0, 1, N))
    cmap_name = base.name + str(N)
    # return base.from_list(cmap_name, color_list, N)
    return LinearSegmentedColormap.from_list(cmap_name, color_list, N)

if __name__ == '__main__':
    #construct tf tensor flow
    Fold = 1
    class_num = 7
    reports = []
    os.environ["CUDA_VISIBLE_DEVICES"] = "1"
    isess = tf.InteractiveSession()
    img_input = tf.placeholder(tf.float32, shape=(512, 512))
    image_4d = tf.expand_dims(img_input, 0)
    image_4d = tf.expand_dims(image_4d, -1)  # add the batch dimension.
    _, logits = SpinePathNet.g_net_graph(image_4d, Fold, batch_size=1, class_num=7, reuse=None, is_training=False,
                                        scope='g_SpinePathNet')
    pred = tf.argmax(tf.nn.softmax(logits), dimension=3)
    init_op = tf.group(tf.global_variables_initializer(),
                       tf.local_variables_initializer())
    isess.run(init_op)
    # checkpoint_dir.
    loader = tf.train.Saver()
    checkpoint_dir = 'tmp/graph/tfmodels_gan_%s/' % Fold
    report_save_dir = 'tmp/graph/tfmodels_gan_%s/generated_reports.txt'% Fold
    ckpt = tf.train.get_checkpoint_state(checkpoint_dir)
    if ckpt and ckpt.model_checkpoint_path:
        # Restores from checkpoint
        loader.restore(isess, ckpt.model_checkpoint_path)
        print("Restored model parameters from {}".format(ckpt))
    else:
        print('No checkpoint file found')
    path = 'datasets/spine_segmentation/spine_segmentation_%s/test/' % Fold
    DIRECTORY_ANNOTATIONS = path + 'Annotations/'
    DIRECTORY_IMAGES = path + 'Dicoms/'
    image_names = sorted(os.listdir(DIRECTORY_IMAGES))  ##files list.
    anno_names = sorted(os.listdir(DIRECTORY_ANNOTATIONS))
    # Read data and fitting in the tf graph.
    for i, j in tqdm.tqdm(enumerate(image_names)):
        image = cdtt.get_image_data_from_dicom(DIRECTORY_IMAGES + j)
        _, mask_class, _, _, _, _ = cdtt.groundtruth_to_mask(DIRECTORY_ANNOTATIONS + anno_names[i])
        predictions = isess.run([pred], feed_dict={img_input: image})
        predictions = np.reshape(predictions, [512, 512])
        #np.save('tmp/graph/tfmodels_gan_1/predict_{0}.npy'.format(i), predictions)
        # generate radiological reports.
        generator = spinal_radiological_reports_generator(predictions, class_num)
        report = generator.report_generation(j.split('.')[0])
        reports.append(report)
    with open(report_save_dir, 'w') as f:
        for item in reports:
            f.write("%s\n" % item)
        #x = generator.summarize_pixel_one_structure("disc")

        # print('              Image                             Ground truth                            Prediction')
        #
        # plt.figure(figsize=(15, 15))
        #
        # # Raw image.
        # ax_1 = plt.subplot(131)
        # ax_1.set_axis_off()
        # divider_1 = make_axes_locatable(ax_1)
        # im_1 = ax_1.imshow(image, cmap='bone')
        # cax_1 = divider_1.append_axes("right", size="5%", pad=0.05)
        # plt.colorbar(im_1, cax=cax_1)
        #
        # # Ground truth.
        # ax_2 = plt.subplot(132)
        # ax_2.set_axis_off()
        # im_2 = plt.imshow(mask_class, cmap=discrete_cmap(class_num, 'nipy_spectral'))  # gist_ncar,CMRmap
        # divider_2 = make_axes_locatable(ax_2)
        # cax_2 = divider_2.append_axes("right", size="5%", pad=0.05)
        # gt_1 = plt.colorbar(im_2, cax=cax_2)
        # # plt.savefig('%sgt_%s.jpg'%(checkpoint_dir,j.split(".")[0]))#format='eps'
        #
        # # Prediction.
        # ax_3 = plt.subplot(133)
        # ax_3.set_axis_off()
        # divider_3 = make_axes_locatable(ax_3)
        # im_3 = ax_3.imshow(predictions, cmap=discrete_cmap(class_num, 'nipy_spectral'))
        # cax_3 = divider_3.append_axes("right", size="5%", pad=0.05)
        # plt.colorbar(im_3, cax=cax_3)
        # plt.savefig('%splot_%s.pdf' % (checkpoint_dir, j.split(".")[0]), bbox_inches='tight', pad_inches=0,
        #             format='pdf',
        #             frameon=False, dpi=600, transparent=True, )
        # # plt.axis('off')
        # plt.show()

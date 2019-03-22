import numpy as np
import scipy
import glob
import dicom
import tqdm
import cv2
import scipy.ndimage as ndimage
import xml.etree.ElementTree as ET
from sklearn.preprocessing import normalize
from matplotlib import pyplot as plt


SPINE_LABELS = {
    'none':(0,'background'),
    'vn':(1, 'Normal Vertebral'),
    'vd':(4, 'Vertebral Deformity'),
    'dn':(2, 'Normal Disc'),
    'dm':(5, 'Mild Gegeneration Disc'),
    'ds':(5, 'Severe Degeneration Disc'),
    'fn':(3, 'Neuro Foraminal Normal'),
    'fs':(6, 'Neuro Foraminal Stenosis'),
    'sv':(0, 'Caudal Vertebra')
    }

def get_image_data_from_dicom(dm, w=512., h=512.):
    """
    input: a dicom file.
    param dm: target width and height.
    return: image data of numpy array.
    """
    dm = dicom.read_file(dm)
    wscale = w/dm.Rows
    hscale = h/dm.Columns
    image_data = np.array(dm.pixel_array)
    image_data = ndimage.interpolation.zoom(image_data, [wscale,hscale])
    return image_data

def get_groundtruth_from_xml(xml):
    labels = []
    labels_text = []
    instance = []
    coordinates_class = []  # The key of this dictionary is the class and values are class' coordinates.
    coordinates_instance = {}  # The key of this dictionary is the class and values are instance' coordinates.
    tree = ET.parse(xml)
    root = tree.getroot()
    rows = root.find('imagesize').find('nrows').text
    columns = root.find('imagesize').find('ncols').text
    shape = [int(rows), int(columns), int(1)]
    masks = np.array([rows, columns])
    for object in root.findall('object'):
        coordinate = []
        if object.find('deleted').text != 1:
            label = object.find('name').text  # class-wise character groundtruth
            label_int = int(SPINE_LABELS[label][0])  # class-wise number groundtruth
            # append to lists
            labels.append(label_int)
            labels_text.append(label.encode('ascii'))

            instance_label_int = int(object.find('id').text)  # instance-wise number groundtruth
            instance.append(instance_label_int)
            polygon = object.find('polygon')
            for pt in polygon.findall('pt'):
                x = int(pt.find('x').text)
                y = int(pt.find('y').text)
                coordinate.append((x, y))
            coordinates_class.append(coordinate)
            coordinates_instance[instance_label_int] = coordinate
    return labels, labels_text, instance, shape, coordinates_class, coordinates_instance

def compute_coordinates(polygon):
    """
    compute the target coordinates of a series of polygons.
    :param polygon: [(x1,y1), {x2, y2),...]]
    :return: [xmin, ymin, xmax, ymax], [xcenter, ycenter]
    """
    x = []
    y = []
    for i in polygon:
        x.append(i[0])
        y.append(i[1])
    xmin = np.min(x)
    xmax = np.max(x)
    ymin = np.min(y)
    ymax = np.max(y)
    xcenter = (xmax+xmin)/2
    ycenter = (ymax+ymin)/2
    return [xmin, ymin, xmax, ymax], [xcenter, ycenter]

def restore_feature(label, structure_number, feature, node_representation):
    if label == 1:
        structure_number[0] += 1#NV: 0, ND: 1, NNF: 2, AV: 3, AD: 4, ANF: 5
        node_representation[0,:] += feature
    elif label == 2:
        structure_number[1] += 1
        node_representation[1, :] += feature
    elif label == 3:
        structure_number[2] += 1
        node_representation[2, :] += feature
    elif label == 4:
        structure_number[3] += 1
        node_representation[3, :] += feature
    elif label == 5:
        structure_number[4] += 1
        node_representation[4, :] += feature
    elif label == 6:
        structure_number[5] += 1
        node_representation[5, :] += feature

def compute_node_repres(node_representation, structure_number, polygons, labels, img_npy):
    """
    Compute the node representation of every structures.
    :param node_representation:
        init features for the six types of spinal structures: NV: 0, ND: 1, NNF: 2, AV: 3, AD: 4, ANF: 5 with size of (6,128)
    :param structure_number: #the number of spinal structures.NV: 0, ND: 1, NNF: 2, AV: 3, AD: 4, ANF: 5
    :param polygons:
    :param labels:
    :param img_npy:
    :return:None
    """
    #sift = cv2.xfeatures2d.SIFT_create()
    #hog = cv2.HOGDescriptor()
    for i, polygon in enumerate(polygons):
        label = labels[i]
        rect_coor, _ = compute_coordinates(polygon)
        xmin = rect_coor[0]
        ymin = rect_coor[1]
        xmax = rect_coor[2]
        ymax = rect_coor[3]

        if ymax - ymin <= 2 or xmax - xmin <= 2 or ymin <= 0 or xmin <= 0:
            continue
        #if label == 3 or label == 6:
        #    structure_patch = img_npy[ymin-10:ymax+10, xmin-10:xmax+10]
        #elif label == 2 or label == 5:
        #    structure_patch = img_npy[ymin - 8:ymax + 8, xmin - 8:xmax + 8]
        #else:
        #    structure_patch = img_npy[ymin - 5:ymax + 5, xmin - 5:xmax + 5]
        #if ymax - ymin <= 2 or xmax - xmin <= 2:
        #    continue


        #cv2.imwrite('patches/structure_{0}_{1}.png'.format(i, label), structure_patch)
        #print structure_patch.shape
        high = ymax - ymin
        width = xmax - xmin
        aspect = (xmax - xmin) / (ymax - ymin)
        selfdesign_feature = np.asarray((high, width, aspect))
        structure_patch = img_npy[ymin:ymax, xmin:xmax]
        #_, histogram = sift.detectAndCompute(structure_patch, None)#TODO conder a new feature extractor.
        histogram = cv2.calcHist([structure_patch], [0], None, [256], [0, 256])
        histogram = np.reshape(histogram, [256])
        #feature = np.concatenate((selfdesign_feature, histogram), axis=0)
        feature = histogram
        if feature is None:
             continue
        restore_feature(label, structure_number, feature, node_representation)

def compute_node_edge(node_edge, polygons, labels):
    """
    Compute the edges between nodes using soft connections (0,1)
    :param node_correlation:
    :param polygons:
    :param labels:
    :return: None
    """
    structure_number =  int(len(labels)/3) #the number of one type structure.
    structure = np.zeros((3,structure_number))# consider a 3*one type structure matrix: [
    s_order = [] # the oder of structures
    for i, polygon in enumerate(polygons):
        _, center_coor = compute_coordinates(polygon)
        s_order.append((labels[i], center_coor[1],))#(y,label)
    def sortSecond(item):
        return item[1]
    s_order.sort(key = sortSecond, reverse = True)

    s_order = s_order[:structure_number]
    for i, s in enumerate(s_order): #TODO consider the logic behand this.
        if s[0] == 1 or s[0] == 4:
            index = structure_number - int(i/3) - 1
            structure[0,index] = s[0]
        elif s[0] == 2 or s[0] == 5:
            index = structure_number - int(i/3) - 1
            structure[1,index] = s[0]
        elif s[0] == 3 or s[0] == 6:
            index = structure_number - int(i/3) - 1
            structure[2,index] = s[0]
    for i in xrange(structure_number):
        ver_current = structure[0, i]
        disc_current = structure[1, i]
        nf = structure[2, i]
        # the class order of node edge is NV: 0, ND: 1, NNF: 2, AV: 3, AD: 4, ANF: 5
        if nf == 6:
            if ver_current == 4:
                node_edge[3, 5] += 1
                node_edge[5, 3] += 1
            else:
                node_edge[0, 5] += 1
                node_edge[5, 0] += 1
            if disc_current == 5:
                node_edge[4, 5] += 1
                node_edge[5, 4] += 1
            else:
                node_edge[1, 5] += 1
                node_edge[5, 1] += 1
        else:
            if ver_current == 4:
                node_edge[3, 2] += 1
                node_edge[2, 3] += 1
            else:
                node_edge[0, 2] += 1
                node_edge[2, 0] += 1
            if disc_current == 5:
                node_edge[4, 2] += 1
                node_edge[2, 4] += 1
            else:
                node_edge[1, 2] += 1
                node_edge[2, 1] += 1
        if ver_current == 4:
            if disc_current == 5:
                node_edge[3, 4] += 1
                node_edge[4, 3] += 1
            else:
                node_edge[3, 1] += 1
                node_edge[1, 3] += 1
        else:
            if disc_current == 5:
                node_edge[0, 4] += 1
                node_edge[4, 0] += 1
            else:
                node_edge[0, 1] += 1
                node_edge[1, 0] += 1
        #TODO: set the coorelation between context discs and vertebrae

def gene_knowledge_graph(anno_dir, data_dir):
    """
    This function is to generate the knowledge graph including nodes and its edges.
    :param anno_dir: the annotation file directions
    :param data_dir: the data file directions
    :return: graph
    """
    node_representation = np.zeros((6, 256))  # init features for the six types of spinal structures:NV: 0, ND: 1, NNF: 2, AV: 3, AD: 4, ANF: 5
    #the class order of node edge is NV: 0, ND: 1, NNF: 2, AV: 3, AD: 4, ANF: 5
    node_edge = np.zeros((6, 6))  # init the edge between the six types of spinal structures.
    structure_number = np.zeros((6)) #the number of spinal structures.NV: 0, ND: 1, NNF: 2, AV: 3, AD: 4, ANF: 5
    anno_filenames = glob.glob(anno_dir)
    for anno_filename in anno_filenames:
        data_filename = data_dir + anno_filename.split("/")[-1].split(".")[0] + '.dcm'
        labels, _, _, _, polygons, _ = get_groundtruth_from_xml(anno_filename)
        #print data_filename
        img_npy = get_image_data_from_dicom(data_filename)
        img_npy = img_npy.astype(np.float32)/img_npy.max()
        img_npy = img_npy*255
        img_npy = img_npy.astype(np.uint8)
        compute_node_edge(node_edge, polygons, labels)
        compute_node_repres(node_representation, structure_number, polygons, labels, img_npy)
    #print node_representation, structure_number
    node_representation = np.divide(node_representation, structure_number[:, None])#TODO
    degree_matrix = np.zeros((6, 6))
    for i in xrange(node_edge.shape[0]):
        degree_matrix[i,i] = np.sum(node_edge[i,:])
    node_edge = node_edge + np.identity(6)
    degree_matrix = np.linalg.inv(scipy.linalg.sqrtm(degree_matrix))
    node_edge = np.matmul(np.matmul(degree_matrix, node_edge), degree_matrix)
    #node_edge = normalize(node_edge, axis=1, norm='l1')
    #node_representation = normalize(node_representation, axis=1, norm='l1')
    knowledge_graph = {'node_representation': node_representation,
                       'node_edge': node_edge}
    return knowledge_graph


if __name__ == '__main__':
    for fold in tqdm.tqdm(xrange(5)):
        fold += 1 #five fold cross validation.
        anno_dir = 'datasets/spine_segmentation/spine_segmentation_{0}/train/Annotations/*.xml'.format(str(fold))
        data_dir = 'datasets/spine_segmentation/spine_segmentation_{0}/train/Dicoms/'.format(str(fold))
        graph_save_dir = 'datasets/spine_segmentation/spine_segmentation_{0}/knowledge_graph.npy'.format(str(fold))
        graph = gene_knowledge_graph(anno_dir, data_dir)
        np.save(graph_save_dir, graph)




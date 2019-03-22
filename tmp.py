import numpy as np
from skimage.measure import label
from skimage.measure import regionprops
import matplotlib.pyplot as plt
for i in xrange(10):
    mapfile = 'tmp/graph/tfmodels_gan_1/predict_{0}.npy'.format(i)
    mapfile = np.load(mapfile)
    spinal_structure = "foramen"
    labels = []
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

    map[map==reserve-1] = reserve
    map[map!=reserve] = 0

    label_map = label(map)
    props = regionprops(label_map)
    spinal_order = []
    for z, j in enumerate(props):
        if j.area >= area_threshold:
            spinal_order.append(z)
    if len(spinal_order) > 5:
        spinal_order = spinal_order[-5:] #reserve the final five structures for the report generation.
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


    #plt.imsave('tmp/graph/tfmodels_gan_1/{0}_map_{1}_{2}.jpg'.format(spinal_structure, i, len(spinal_order)), map)


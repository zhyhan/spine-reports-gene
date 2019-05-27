import numpy
from pyswip import Prolog


def relation_learn(disc_conditions, foramen_conditions, vertebrae_conditions):
    """
    :param disc_conditions: a list of prediction labels with length of five, such as [0,0,0,1,1]
    :param foramen_conditions: components
    :param vertebrae_conditions: same before
    :return: a list of impact results [who impact NFS, impact result];
             a list of labels of foramen from unsupervised learning.
    """
    prolog = Prolog()
    prolog.consult("logic/relation_learn.pl")
    impact = []
    un_label_NF = [] #unsupervised label
    loc = ['L1', 'L2', 'L3', 'L4', 'L5'] #TODO consider the location information.
    for i, disc_cond in enumerate(disc_conditions):
        location = loc[i]
        if disc_cond == 1:
            disc_cond = 'ldd' #Lumbar disc deformation.
        else:
            disc_cond = 'nld' #Normal lumbar disc
        if vertebrae_conditions[i] == 1:
            vertebra_cond = 'lvd' #Lumbar vertebral deformation.
        else:
            vertebra_cond = 'nlv' #Normal Lumbar vertebral.
        if foramen_conditions[i] == 1:
            foramen_cond = 'lnfs' #Lumbar neural formal stenosis
        else:
            foramen_cond = 'nlnf'#Normal Lumbar formal

        q = prolog.query("impact(%s,%s,%s)"%(disc_cond,vertebra_cond,foramen_cond))
        if bool(list(q)) is True:
            impact.append(1)
        else:
            impact.append(0)

        p = prolog.query("impact(%s,%s,X)"%(disc_cond,vertebra_cond))
        if len(list(p))>0:
            un_label_NF.append(1)
        else:
            un_label_NF.append(0)

    return impact, un_label_NF

if __name__ == '__main__':
    #1. find the cause of LNFS.
    #2. find the NF label
    #3. discover the correlation.

    disc_conditions = [1,0,1,0,1]
    foramen_conditions = [1,0,1,0,1]
    vertebrae_conditions = [0,1,0,0,1]
    impact, un_label_NF = relation_learn(disc_conditions, foramen_conditions, vertebrae_conditions)

    print(impact, un_label_NF)
    # prolog = Prolog()
    # prolog.consult("relation_learn.pl")
    # q = prolog.query("impact(ld,_,lnfs)")
    # print bool(list(q))
    # for i, z in enumerate(q):
    #      print z




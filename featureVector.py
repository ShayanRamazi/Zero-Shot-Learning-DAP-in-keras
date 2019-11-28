"""
محاسبه بردار ویژگی برای حیوانات
Compute F.V. from Vgg19

"""

import os, os.path
import numpy as np
import pickle as cPickle
import bz2
import concatenateFV as cfv
import math
import random
def writeInFile(obj,filename):
    f = bz2.BZ2File(filename, 'wb')
    cPickle.dump(obj, f)
    f.close()

def createFeaturesVector(path,feat_shape=4096):
    trainclasses = cfv.loadstr('./Classes/trainclasses.txt')
    feat_shape = 4096
    subfolders = [x[0] for x in os.walk(path)][1:]
    features=''
    featuresTest=''
    className=''

    for i,subfolder in enumerate(subfolders):
        animal = subfolder.split('/')[-1:][0]
        nb_files = len([name for name in os.listdir(subfolder) if os.path.isfile(os.path.join(subfolder, name))])
        tempi = 0
        l=random.sample(range(1,nb_files+1),int(math.ceil(nb_files*0.3)))
        for j in range(1,nb_files+1):
            img_feat_path = subfolder + '/' + animal + '_{:04d}'.format(j) + '.txt'

            if j==1:
                features = np.loadtxt(img_feat_path, comments="#", delimiter=",", unpack=False)
                features = features.reshape(((feat_shape,1)))
            elif j in l and (animal in trainclasses) and tempi==0 :
                tempi+=1
                featuresTest = np.loadtxt(img_feat_path, comments="#", delimiter=",", unpack=False)
                featuresTest = featuresTest.reshape(((feat_shape, 1)))
            elif j in l and (animal in trainclasses):
                tempi += 1
                tmp_features = np.loadtxt(img_feat_path, comments="#", delimiter=",", unpack=False)
                tmp_features = tmp_features.reshape(((feat_shape, 1)))
                featuresTest = np.concatenate((featuresTest, tmp_features), axis=1)
            else:
                tmp_features = np.loadtxt(img_feat_path, comments="#", delimiter=",", unpack=False)
                tmp_features = tmp_features.reshape(((feat_shape,1)))
                features = np.concatenate((features,tmp_features), axis=1)

        try:
            os.stat('feat/')
        except:
            os.mkdir('feat/')
        try:
            os.stat('featx/')
        except:
            os.mkdir('featx/')

        try:
            os.stat('Classes/')
        except:
            os.mkdir('Classes/')

        picklefile = 'feat/featuresVGG19_' + animal + '.pic.bz2'
        print ("Create ",animal," features vector ")
        writeInFile(features, picklefile)

        picklefileTest = 'featx/featuresVGG19_' + animal + '.pic.bz2'
        print ("Create ",animal," features vector ")
        writeInFile(featuresTest, picklefileTest)

        f = open("Classes/num.txt", "a+")
        f.write(animal+" %d\n" % (tempi))
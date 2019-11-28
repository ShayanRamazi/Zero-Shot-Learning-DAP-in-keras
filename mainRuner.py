"""
 main function
 auther:sh.ramazi
"""
import featureVector as fv
import concatenateFV as cfv
import directAttributePrediction as dap
import DAP_eval as dapEval
import numpy as np
#main
if __name__ == '__main__':
    # fv.createFeaturesVector("./Features/vgg19/")
    # # # # # Training classes
    #print ('#### Concatenating training data....')
    #trainclasses = cfv.loadstr('./Classes/trainclasses.txt')
    # cfv.concatenate_set_features(trainclasses, 'train')
    # # # # #
    #print ('#### Concatenating validation data....')
    #trainclasses = cfv.loadstr('./Classes/trainclasses.txt')
    #cfv.concatenate_set_featuresx(trainclasses, 'validation')
    # # Test classes
    # print ('#### Concatenatin# g test data....')
    # testclasses = cfv.loadstr('./Classes/testclasses.txt')
    # cfv.concatenate_set_features(testclasses, 'test')

    # # DAP NN
    # dap.DirectAttributePrediction()
    # # #
    # # # #DAP_eval
    # attributepattern = './DAP_binary/probabilities_' + 'NN'
    # confusion, prob, L = dapEval.evaluate(0, 10, attributepattern)
    # # # dapEval.plot_confusion(confusion, 'NN')
    # # # dapEval.plot_roc(prob, L, 'NN')
    # # # dapEval.plot_attAUC(L, attributepattern, 'NN')
    # print("Mean class accuracy unseen %g" % np.mean(np.diag(confusion) * 100))
    attributepattern = './DAP_binary/xprobabilities_' + 'NN'
    confusion, prob, L = dapEval.evaluate_(0, 40, attributepattern)
    confusion[np.isnan(confusion)] = 0
    print("Mean class accuracy seen %g" % np.mean(np.diag(np.nan_to_num(confusion[:,:])) * 100))
#IMPORT LIBRARIES
import numpy as np
import pandas as pd
import scipy as sp
import matplotlib.pyplot as plt
import pvml



def show_spectrogram(features, labels, classes, word, cbar = True):
    '''Returns the spectrogram of the first occurrence of word in features matrix'''
    
    #find row of word in features
    row_index = np.where(classes == word)[0][0] 
    row_index = np.where(labels == row_index)[0][0]
    print('row index = ', row_index)
    
    #reshape row into spectrogram and show it
    spectro = features[row_index,:].reshape(20, 80)
    plt.imshow(spectro, cmap = 'hot', aspect='auto')
    if cbar is True:
        plt.colorbar()
    plt.ylabel('Frequency')
    plt.xlabel('Time')
    plt.title('Spectrogram of ' + word)



def show_spectrogram_multiple(features, labels, classes, words):
    '''Returns the spectrogram of the first occurrence of each word of words in the features matrix'''
    
    #define colorbar
    sm = plt.cm.ScalarMappable(cmap='hot')
    sm.set_array(features)
    
    fig, axs = plt.subplots(1, len(words), figsize=(20, 5))
    
    for w, i in zip(words, range(len(axs))):
        
        #find row of w in features
        row_index = np.where(classes == w)[0][0]
        row_index = np.where(labels == row_index)[0][0]
        print('row index of %s = ' %w, row_index)
        
        #reshape row into spectrogram and show it
        spectro = features[row_index,:].reshape(20, 80)
        axs[i].imshow(spectro, cmap = 'hot', aspect='auto')
        axs[i].set_title('word: %s' %w)
        
    fig.suptitle('Spectrograms')
    fig.text(0.5, 0.02, 'Time', ha='center')
    fig.text(0.08, 0.5, 'Frequency', va='center', rotation='vertical')
    fig.colorbar(sm, ax=axs)
    plt.show()
    

def mean_var_normalize(train_features, test_features):
    '''returns normalized train and test features using mean-variance normalization'''
    u = train_features.mean(0)
    sigma = train_features.std(0)
    train_features = (train_features - u) / sigma
    test_features = (test_features - u) / sigma

    return train_features, test_features


def min_max_normalize(train_features, test_features):
    '''returns normalized train and test features using max-min normalization'''
    min = train_features.min(0)
    max = train_features.max(0)
    train_features = (train_features - min) / (max - min)
    test_features = (test_features - min) / (max - min)
    
    return train_features, test_features
    
    
def max_abs_normalize(train_features, test_features):
    '''returns normalized train and test features using max-abs normalization'''
    max = np.abs(train_features).max(0)
    train_features = train_features / max
    test_features = test_features / max
    
    return train_features, test_features

def whitening_normalize(Xtrain , Xtest):
    mu = Xtrain.mean(0)
    sigma = np.cov(Xtrain.T)
    evals , evecs = np.linalg.eigh(sigma) 
    w = evecs / np.sqrt(evals)
    Xtrain = (Xtrain - mu) @ w 
    Xtest = (Xtest - mu) @ w 
    return Xtrain , Xtest
    
def accuracy(net, X, Y):
    '''returns the accuracy of the network on the given data X and labels Y'''
    labels, probs = net.inference(X)
    acc = (labels == Y).mean()
    return acc * 100


def confusion_matrix(Y, predictions, labels, show = False, rnorm = True):
    '''Displays the confusione matrix. If rnorm is True, the values are normalized respect to the actual number of samples in each class'''
    classes = Y.max() + 1
    cm = np.empty((classes, classes))
    for klass in range(classes):
        sel = (Y == klass).nonzero()
        counts = np.bincount(predictions[sel], minlength=classes)
        if rnorm is False:
            cm[klass, :] = counts
        else:
            cm[klass, :] = 100 * counts / max(1, counts.sum())
        
    if show is True:
        plt.figure(3, figsize=(20, 20))
        plt.clf()
        plt.xticks(range(classes), labels, rotation=45)
        plt.yticks(range(classes), labels)
        plt.imshow(cm, vmin=0, vmax=100, cmap='Blues')
        for i in range(classes):
            for j in range(classes):
                txt = "{:.1f}".format(cm[i, j], ha="center", va="center")
                col = ("black" if cm[i, j] < 75 else "white")
                plt.text(j - 0.25, i, txt, color=col)
        plt.title("Confusion Matrix") 
    return cm


def likely_misclassified(occurrs, cm, classes, n, show = False):
    '''Returns the n classes for which the delta between 100 and the percentage of correct classified is the largest'''
    deltas = []
    for i in range(35):
        delta = 100-cm[i,i]
        deltas.append(delta)
    indexes = np.argsort(deltas)
    deltas = np.array(deltas)
    indexes = indexes[-n:]
    arr = np.array([classes[indexes], np.floor(deltas[indexes]), occurrs[indexes]])
    if show is True:
        plt.figure(1)
        plt.vlines(arr[0,:], ymin = 0, ymax = arr[1,:])
        #plt.gca().invert_yaxis()
        plt.title('Most Likely Misclassified')
        plt.ylabel('Delta: 100 - correct (%)')
        plt.xticks(rotation = 90)
        plt.show(1)
    return arr


def wrong_classes(cm, classes):
    '''For each class returns the class respect to which it is more likely to be exchanged (with labels and with values)'''
    actuals = [] 
    wrongs = []
    vals = []
    for i in range(35):
        c = np.argmax(cm[i,:])
        if c == i:
            cm[i,c]=0
            c = np.argmax(cm[i,:])
        value = cm[i,c]
        actuals.append(i)
        wrongs.append(c)
        vals.append(value)
    out_valued = np.array([actuals, wrongs, vals])
    out_labelled = np.array([classes[actuals], classes[wrongs]])
    return out_labelled, out_valued


def show_weights_single(network, classes, klass):
    '''Returns an image with the weights of the first layer of the network for the class klass'''
    w = network.weights[0]
    maxval = np.abs(w).max()
    plt.imshow(w[:,klass].reshape(20,80), cmap = 'seismic', vmin = -maxval, vmax = maxval, aspect='auto')
    plt.colorbar()
    plt.title(classes[klass])
    
    
def show_weights_multiple(network, classes):
    '''Returns an image with the weights of the first layer of the network for each class'''
    w = network.weights[0]
    maxval = np.abs(w).max()
    plt.figure(figsize=(20,10))
    for klass in range(35):
        plt.subplot(5, 7, klass +1 )
        plt.imshow(w[:,klass].reshape(20,80), cmap = 'seismic', vmin = -maxval, vmax = maxval)
        plt.title(classes[klass])
    plt.show()
    
    
def n_misclassified(cm, classes, to_show = 0):
    '''Returns the words the model chosen wrongly as prediction most of times'''
    arr = np.empty((35))
    for j in range(35):
        tot = max(cm[:,j].sum() - cm[j,j], 0)
        arr[j] = tot
    arr = np.round(arr / arr.sum() * 100, 2)

    if to_show != 0:
        indexes = np.argsort(arr)
        indexes = indexes[-to_show:]
        tmp = np.array([classes[indexes], arr[indexes]])
        
        plt.figure()
        plt.vlines(x = tmp[0,:], ymin = 0, ymax = tmp[1,:])
        plt.xticks(rotation = 90)
        plt.title('%d Most Chosen Words' % to_show)
        plt.ylabel('got wrong (%)')
        
    return arr


def sorting_list(dim, n):
    numbers = []
    for i in range(dim):
        numbers.append(i)
    return numbers*n



############ DEFINITION OF TEMPLATES ############


#template used to compare the batches
def train_template(net, train_X, train_Y, test_X, test_Y, batch_size, n_epochs, moment=0, lambda_val=0.00001):
    n_iter = train_X.shape[0]//batch_size
    train_accs = []
    test_accs = []
    epochs = []
    
    for epoch in range(n_epochs):
        net.train(train_X, train_Y, 1e-4, steps=n_iter, batch=batch_size, momentum = moment, lambda_=lambda_val)
        if epoch % 10 == 0:
            train_acc = accuracy(net, train_X, train_Y)
            test_acc = accuracy(net, test_X, test_Y)
            print(epoch, train_acc, test_acc)
            train_accs.append(train_acc)
            test_accs.append(test_acc)
            epochs.append(epoch)
    return train_accs, test_accs
            
            
            
def compare_batches(train_X, train_Y, test_X, test_Y, batch_sizes, n_epochs = 100):

    for batch_size in batch_sizes:
        net = pvml.MLP([1600, 35])
        train_accs, test_accs = train_template(net, train_X, train_Y, test_X, test_Y, batch_size, n_epochs)
    
        with open('../results/accuracies.txt', 'a') as f:
            for train_acc, test_acc in zip(train_accs, test_accs):
                f.write(f"{batch_size}\t{train_acc}\t{test_acc}\n")
                 

#template used to compare the neural network architectures (batch fixed)
def compare_arch(train_X, train_Y, test_X, test_Y, batch_size, n_hidden, width, net, moment = 0, n_epochs = 100, store = False, lambda_val = 0.001):
    
    train_accs, test_accs = train_template(net, train_X, train_Y, test_X, test_Y, batch_size, n_epochs, moment, lambda_val)

    #store the results computing the number of parameters
    with open('../results/accuracies_14.txt', 'a') as f:
        for train_acc, test_acc in zip(train_accs, test_accs):
            if len(width)==1:
                N_parameters = (1600+1)*width[0] + width[0]*(35+1)
                f.write(f"{n_hidden}\t{width[0]}\t{N_parameters}\t{train_acc}\t{test_acc}\n")
            elif len(width)==2:
                N_parameters = (1600+1)*width[0] + width[0]*(width[1]+1) + width[1]*(35+1)
                f.write(f"{n_hidden}\t{width[0]}x{width[1]}\t{N_parameters}\t{train_acc}\t{test_acc}\n")
            elif len(width)==3:
                N_parameters = (1600+1)*width[0] + width[0]*(width[1]+1) + width[1]*(width[2]+1) + width[2]*(35+1)
                f.write(f"{n_hidden}\t{width[0]}x{width[1]}x{width[2]}\t{N_parameters}\t{train_acc}\t{test_acc}\n")
            elif len(width)==4:
                N_parameters = (1600+1)*width[0] + width[0]*(width[1]+1) + width[1]*(width[2]+1) + width[2]*(width[3]+1) + width[3]*(35+1)
                f.write(f"{n_hidden}\t{width[0]}x{width[1]}x{width[2]}\t{N_parameters}\t{train_acc}\t{test_acc}\n")
            else:
                N_parameters = (1600+1)*35
                f.write(f"{n_hidden}\t{0}\t{N_parameters}\t{train_acc}\t{test_acc}\n")
                
    if store is True: 
        net.save('MLP_trained.npz')
        
        
#template used to compare the normalization methods (batch and architecture fixed)
def compare_norm(train_X, train_Y, test_X, test_Y, name, batch_size = 50, n_hidden = 3, width=[256,128,56], moment = 0.9, n_epochs = 100, store = False):
    net = pvml.MLP([1600, 256, 128, 56, 35])
    train_accs, test_accs = train_template(net, train_X, train_Y, test_X, test_Y, batch_size, n_epochs, moment)

    with open('../results/accuracies_norm.txt', 'a') as f:
        for train_acc, test_acc in zip(train_accs, test_accs):
            N_parameters = (1600+1)*width[0] + width[0]*(width[1]+1) + width[1]*(width[2]+1) + width[2]*(35+1)
            f.write(f"{n_hidden}\t{width[0]}x{width[1]}x{width[2]}\t{N_parameters}\t{train_acc}\t{test_acc}\n")
        
                
    if store is True: 
        net.save('MLP_' + name + '_trained.npz')
        
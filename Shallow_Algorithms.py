#!/usr/bin/env python
# coding: utf-8

# In[ ]:


import os
import pandas as pd
import numpy as np
import time
from scipy import stats
from sklearn import preprocessing
from sklearn.model_selection import train_test_split
from sklearn import svm
from sklearn.metrics import accuracy_score
from sklearn.metrics import classification_report
from sklearn.metrics import confusion_matrix
from sklearn import svm
from sklearn.neural_network import MLPClassifier
from sklearn.model_selection import KFold
from sklearn import tree
from sklearn.ensemble import RandomForestClassifier
from sklearn.naive_bayes import GaussianNB 
import itertools
from matplotlib import pyplot as plt
from sklearn.decomposition import PCA
from sklearn.neighbors import KNeighborsClassifier
from sklearn.manifold import TSNE
import signal
import scipy 
from sklearn.svm import SVC 
from sklearn.model_selection import ShuffleSplit
import matplotlib as mpl
from matplotlib.colors import ListedColormap
from sklearn.discriminant_analysis import LinearDiscriminantAnalysis
from sklearn.preprocessing import LabelEncoder
import seaborn as sns


# In[ ]:


mpl.rcParams['lines.linewidth'] = 4
mpl.rcParams['lines.color'] = 'r'
mpl.rcParams['font.weight'] = 200
plt.style.use('seaborn-whitegrid')
plt.rc('figure',figsize=(11,6))
mpl.axes.Axes.annotate
mpl.rcParams['axes.labelsize'] = 20
mpl.rcParams['axes.linewidth'] = 4
mpl.rcParams['xtick.labelsize'] = 20
mpl.rcParams['ytick.labelsize'] = 20
mpl.rcParams['axes.edgecolor'] = 'black'
mpl.rcParams['axes.titlesize'] = 20
mpl.rcParams['legend.fontsize'] = 14


# In[ ]:


def plot_confusion_matrix (cm, classes, normalize=False,title='Confusion matrix',cmap=plt.cm.Greens):
    if normalize:
        cm = cm.astype('float') / cm.sum(axis=1)[:, np.newaxis]

    plt.figure(figsize=(10,10))
    plt.grid(b=False)    
    plt.imshow(cm, interpolation='nearest', cmap=cmap)
    plt.title(title)
    plt.colorbar()
    tick_marks = np.arange(len(classes))
    plt.xticks(tick_marks, classes, rotation=90)
    plt.yticks(tick_marks, classes)

    fmt = '.1f' if normalize else 'd'
    thresh = cm.max() / 2.
    for i, j in itertools.product(range(cm.shape[0]), range(cm.shape[1])):
        plt.text(j, i, format(cm[i, j], fmt),
                 horizontalalignment="center",
                 color="white" if cm[i, j] > thresh else "black")

    plt.tight_layout()
    plt.ylabel('True label')
    plt.xlabel('Predicted label')


# # Upload your Dataset

# In[ ]:


training_label.shape


# # Median Filter

# In[ ]:


training_data = scipy.signal.medfilt(training_data, kernel_size=3)


# In[ ]:


Y=training_label.reshape(training_label.shape[0])


# In[ ]:


#training_Label = pd.Series(Y)


# In[ ]:


#training_Label.value_counts()


# # Visualization - No. of Datapoints

# In[ ]:


fig = plt.figure(figsize=(9,6))
training_Label.value_counts().plot(kind='bar',fontsize=14,width=0.8)
plt.plot(linewidth=10)
plt.title('Activity',fontsize=18,fontweight='bold')
plt.ylim(0,12000)
plt.xticks(rotation=90)
plt.ylabel('Data Instances',fontsize=14,fontweight='bold')
plt.xlabel('Classes',fontsize=14,fontweight='bold')
fig.savefig('datapoint.jpg', bbox_inches='tight', pad_inches=0)


# In[ ]:


fig = plt.figure(figsize=(12,9))
ax = training_Label.value_counts().plot(kind='bar',fontsize=14,width=0.7);
plt.plot(linewidth=30)
plt.title('Activity',fontsize=18,fontweight='bold')
plt.ylim(0,12000)
plt.xticks(rotation=90)
plt.ylabel('Data Instances',fontsize=14,fontweight='bold')
plt.xlabel('Classes',fontsize=14,fontweight='bold')
totals = []
for i in ax.patches:
    totals.append(i.get_height())

total = sum(totals)

for i in ax.patches:
    ax.text(i.get_x()-.04, i.get_height()+80, 
            str(round((i.get_height()/total)*100, 1))+'%', fontsize=15,fontweight='bold',
                color='dimgrey') 
fig.savefig('datapoint.jpg', bbox_inches='tight', pad_inches=0)    


# # Preprocessing

# In[ ]:


label_encoder = LabelEncoder()
integer_encoded = label_encoder.fit_transform(training_label)


# In[ ]:


y = label_encoder.transform(training_label)
y


# In[ ]:


Y = label_encoder.transform(testing_label) 
Y


# In[ ]:


X = training_data.reshape(training_data.shape[0],-1)


# In[ ]:


x = testing_data.reshape(testing_data.shape[0],-1)


# In[ ]:


X_train, X_test, y_train, y_test = train_test_split(X,y ,test_size=0.2, random_state=3)


# In[ ]:


accuracy_scores = np.zeros(4)


# In[ ]:


target_names = ['Forehand Service','Backhand Service','Clear Lob Overhead Forehand','Clear Lob Overhead Backhand','Clear Lob Underarm Forehand',
         'Clear Lob Underarm Backhand','Net Shot Underarm Forehand','Net Shot Underarm Backhand','Drop Shot Overhead Forehand','Drop Shot Overhead Backhand',
         'Smash Overhead Forehand','Smash Overhead Backhand']


# # LDA Feature Extraction

# In[ ]:


lda = LinearDiscriminantAnalysis(n_components=13)

# run an LDA and use it to transform the features
X_lda = lda.fit(X, y).transform(X)
print('Original number of features:', X.shape[1])
print('Reduced number of features:', X_lda.shape[1])


# In[ ]:


lda = LinearDiscriminantAnalysis(n_components=13)

# run an LDA and use it to transform the features
x_lda = lda.fit(x, Y).transform(x)
print('Original number of features:', x.shape[1])
print('Reduced number of features:', x_lda.shape[1])


# In[ ]:


X_lda.shape


# In[ ]:


#label_dict = {1: 'Setosa', 2: 'Versicolor', 3:'Virginica'}

label_dict=  {1 :  'Forehand Service' , 2: 'Backhand Service' , 3:  'Clear Lob Overhead Forehand', 4: 'Clear Lob Overhead Backhand' ,5 :'Clear Lob Underarm Forehand',
          6 : 'Clear Lob Underarm Backhand' , 7 : 'Net Shot Underarm Forehand', 8: 'Net Shot Underarm Backhand', 9:  'Drop Shot Overhead Forehand' , 10 :'Drop Shot Overhead Backhand',
          11 : 'Smash Overhead Forehand' , 12 : 'Smash Overhead Backhand' }


# In[ ]:


def plot_step_lda():
    fig = plt.figure(figsize=(24,10))
    plt.style.use('seaborn-whitegrid')
    plt.rcParams['axes.edgecolor'] = 'black'
    mpl.rcParams['lines.linewidth'] = 5
    #ax = plt.subplot(111)
    for label,marker,color in zip(
        range(1,13),('^', 's', 'o','.','8','p','h','H','D','X','d','*'),('blue', 'red', 'green','black','yellow', 'cyan', 'green','black','lightblue', 'lightsalmon', 'lightgreen','magenta')):
        plt.scatter(x=x_lda[:,0].real[Y == label],
                y=x_lda[:,1].real[Y == label],
                marker=marker,
                color=color,
                alpha=0.9,
                label=label_dict[label]
                )

    plt.xlabel('LD1', fontsize=22,fontweight='bold')
    plt.ylabel('LD2', fontsize=22,fontweight='bold')

    leg = plt.legend(loc='upper right', fancybox=True, prop={'weight':'bold','size': 10})
    leg.get_frame().set_alpha(0.5)
    #plt.title('LDA: Badminton dataset projection onto the linear discriminants',fontsize=15,  fontweight='bold')

    # hide axis ticks
    plt.tick_params(  bottom="off", top="off",  
            labelbottom="on", left="off", right="off", labelleft="on",labelsize=21)

    # remove axis spines
    #ax.spines["top"].set_visible(False)  
    #ax.spines["right"].set_visible(False)
    #ax.spines["bottom"].set_visible(False)
    #ax.spines["left"].set_visible(False)    
    #fig.savefig('LDA.jpg', bbox_inches='tight', pad_inches=0)
    #plt.grid()
    plt.tight_layout
    plt.show()

plot_step_lda()


# In[ ]:


def plot_step_lda():
    fig = plt.figure(figsize=(24,10))
    plt.style.use('seaborn-whitegrid')
    plt.rcParams['axes.edgecolor'] = 'black'
    mpl.rcParams['lines.linewidth'] = 5
    #ax = plt.subplot(111)
    for label,marker,color in zip(
        range(1,13),('^', 's', 'o','.','8','p','h','H','D','X','d','*'),('blue', 'red', 'green','black','yellow', 'cyan', 'green','black','lightblue', 'lightsalmon', 'lightgreen','magenta')):
        plt.scatter(x=X_lda[:,0].real[y == label],
                y=X_lda[:,1].real[y == label],
                marker=marker,
                color=color,
                alpha=0.9,
                label=label_dict[label]
                )

    plt.xlabel('LD1', fontsize=22,fontweight='bold')
    plt.ylabel('LD2', fontsize=22,fontweight='bold')

    leg = plt.legend(loc='upper right', fancybox=True, prop={'weight':'bold','size': 10})
    leg.get_frame().set_alpha(0.5)
    #plt.title('LDA: Badminton dataset projection onto the linear discriminants',fontsize=15,  fontweight='bold')

    # hide axis ticks
    plt.tick_params(  bottom="off", top="off",  
            labelbottom="on", left="off", right="off", labelleft="on",labelsize=21)

    # remove axis spines
    #ax.spines["top"].set_visible(False)  
    #ax.spines["right"].set_visible(False)
    #ax.spines["bottom"].set_visible(False)
    #ax.spines["left"].set_visible(False)    
    #fig.savefig('LDA.jpg', bbox_inches='tight', pad_inches=0)
    #plt.grid()
    plt.tight_layout
    plt.show()

plot_step_lda()


# In[ ]:


X_Reduced, X_Test_Reduced, Y_Reduced, Y_Test_Reduced = train_test_split(X_lda, y, 
                                                                        test_size=0.2, random_state=3)


# In[ ]:


lda = LinearDiscriminantAnalysis().fit(X_Reduced,Y_Reduced)
predictionlda = lda.predict(X_Test_Reduced)
accuracy_scores[0] = accuracy_score(Y_Test_Reduced, predictionlda)*100
print(' LDA Accuracy: {}%'.format(accuracy_scores[0]))

cm_cmap=plt.cm.Greens
cm = metrics.confusion_matrix(Y_Test_Reduced, predictionlda)
plt.figure(figsize=(10,10))
plt.grid(b=True)
plot_confusion_matrix(cm, classes=target_names, normalize=True, title='Normalized confusion matrix', cmap = cm_cmap)
report = metrics.classification_report(Y_Test_Reduced, predictionlda, target_names=target_names)
#df = pd.DataFrame(report).transpose()
#df.to_csv('Classification_Report_DT_Sreeni_Dataset.csv')
#print(' Decision Tree Classifier Accuracy: {}%'.format(accuracy_scores[0]))
#plt.savefig('Confusion_Matrix_LDA_Participant_Hand_and_Palm_Dataset',transparent=False, pad_inches=10)
print(report)


clf =  MLPClassifier().fit(X_Reduced,Y_Reduced)
prediction = clf.predict(X_Test_Reduced)
accuracy_scores[0] = accuracy_score(Y_Test_Reduced, prediction)*100
cm = metrics.confusion_matrix(Y_Test_Reduced, prediction)
plt.figure(figsize=(8,8))
plt.grid(b=False)
plot_confusion_matrix(cm, classes=target_names, normalize=True, title='Normalized confusion matrix', cmap = cm_cmap)
report = metrics.classification_report(Y_Test_Reduced, prediction, target_names=target_names)
#df = pd.DataFrame(report).transpose()
#df.to_csv('Classification_Report_MLP_Sreeni_Dataset.csv')
print(' MLPClassifier Accuracy: {}%'.format(accuracy_scores[0]))
#cm.numpy.genfromtxt('C:/localpath/test.csv', delimiter=',')
#plt.savefig('Confusion_Matrix_MLP_Participant_Hand_and_Palm_Dataset',transparent=True, bbox_inches='tight', pad_inches=0)
print(report)

gnb = GaussianNB()
clf =  gnb.fit(X_Reduced,Y_Reduced)
prediction = clf.predict(X_Test_Reduced)
accuracy_scores[0] = accuracy_score(Y_Test_Reduced, prediction)*100
cm = metrics.confusion_matrix(Y_Test_Reduced,prediction)
plt.figure(figsize=(8,8))
plt.grid(b=False)
plot_confusion_matrix(cm, classes=target_names, normalize=True, title='Normalized confusion matrix', cmap = cm_cmap)
report = metrics.classification_report(Y_Test_Reduced, prediction, target_names=target_names)
#df = pd.DataFrame(report).transpose()
#df.to_csv('Classification_Report_NB_Sreeni_Dataset.csv')
print(' Naive Bayes Classifier Accuracy : {}%'.format(accuracy_scores[0]))
#plt.savefig('Confusion_Matrix_Naive_Bayes_Participant_Hand_and_Palm_Dataset',transparent=True, bbox_inches='tight', pad_inches=0)
print(report)


clf = RandomForestClassifier().fit(X_Reduced,Y_Reduced)
prediction = clf.predict(X_Test_Reduced)
accuracy_scores[0] = accuracy_score(Y_Test_Reduced, prediction)*100
cm = metrics.confusion_matrix(Y_Test_Reduced, prediction)
plt.figure(figsize=(8,8))
plt.grid(b=False)
plot_confusion_matrix(cm, classes=target_names, normalize=True, title='Normalized confusion matrix', cmap = cm_cmap)
report = metrics.classification_report(Y_Test_Reduced, prediction, target_names=target_names)
#df = pd.DataFrame(report).transpose()
#df.to_csv('Classification_Report_RF_Sreeni_Dataset.csv')
print('Random Forest Classifier Accuracy: {}%'.format(accuracy_scores[0]))
#plt.savefig('Confusion_Matrix_RF_Participant_Hand_and_Palm_Dataset',transparent=True, bbox_inches='tight', pad_inches=0)
print(report)



# # PCA Feature Extraction

# In[ ]:


pca = PCA(n_components=13)
X_pca = pca.fit_transform(X)


# In[ ]:


pca = PCA(n_components=13)
x_pca = pca.fit_transform(x)


# In[ ]:


def plot_pca():
    fig = plt.figure(figsize=(24,10))
    plt.style.use('seaborn-whitegrid')
    plt.rcParams['axes.edgecolor'] = 'black'
    mpl.rcParams['lines.linewidth'] = 5
    #ax = plt.subplot(111)
    for label,marker,color in zip(
        range(1,13),('^', 's', 'o','.','8','p','h','H','D','X','d','*'),('blue', 'red', 'green','black','yellow', 'cyan', 'green','black','lightblue', 'lightsalmon', 'lightgreen','magenta')):

        plt.scatter(x=X_pca[:,0][y == label],
                y=X_pca[:,1][y == label],
                marker=marker,
                color=color,
                alpha=0.9,
                label=label_dict[label]
                )

    plt.xlabel('PC1',fontsize=22,fontweight='bold')
    plt.ylabel('PC2',fontsize=22,fontweight='bold')

    leg = plt.legend(loc='upper left', fancybox=True , prop={'weight':'bold','size': 10})
    leg.get_frame().set_alpha(0.5)
    #plt.title('PCA: Badminton dataset projection onto the first 13 principal components')

    # hide axis ticks
    plt.tick_params(axis="both", which="both", bottom="off", top="off",  
            labelbottom="on", left="off", right="off", labelleft="on" , labelsize='21')

    # remove axis spines
    #ax.spines["top"].set_visible(False)  
    #ax.spines["right"].set_visible(False)
    #ax.spines["bottom"].set_visible(False)
    #ax.spines["left"].set_visible(False)    

    plt.tight_layout
    #plt.grid()
    #fig.savefig('PCA.jpg', bbox_inches='tight', pad_inches=0)
    plt.show()
plot_pca()    


# In[ ]:


def plot_pca():
    fig = plt.figure(figsize=(24,10))
    plt.style.use('seaborn-whitegrid')
    plt.rcParams['axes.edgecolor'] = 'black'
    mpl.rcParams['lines.linewidth'] = 5
    #ax = plt.subplot(111)
    for label,marker,color in zip(
        range(1,13),('^', 's', 'o','.','8','p','h','H','D','X','d','*'),('blue', 'red', 'green','black','yellow', 'cyan', 'green','black','lightblue', 'lightsalmon', 'lightgreen','magenta')):

        plt.scatter(x=x_pca[:,0][Y == label],
                y=x_pca[:,1][Y == label],
                marker=marker,
                color=color,
                alpha=0.9,
                label=label_dict[label]
                )

    plt.xlabel('PC1',fontsize=22,fontweight='bold')
    plt.ylabel('PC2',fontsize=22,fontweight='bold')

    leg = plt.legend(loc='upper left', fancybox=True , prop={'weight':'bold','size': 10})
    leg.get_frame().set_alpha(0.5)
    #plt.title('PCA: Badminton dataset projection onto the first 13 principal components')

    # hide axis ticks
    plt.tick_params(axis="both", which="both", bottom="off", top="off",  
            labelbottom="on", left="off", right="off", labelleft="on" , labelsize='21')

    # remove axis spines
    #ax.spines["top"].set_visible(False)  
    #ax.spines["right"].set_visible(False)
    #ax.spines["bottom"].set_visible(False)
    #ax.spines["left"].set_visible(False)    

    plt.tight_layout
    #plt.grid()
    #fig.savefig('PCA.jpg', bbox_inches='tight', pad_inches=0)
    plt.show()
plot_pca()    


# In[ ]:


plot_pca()
plot_step_lda()


# In[ ]:


pca.explained_variance_ratio_


# # Classifiers

# In[ ]:


cm_cmap=plt.cm.Greens

clf =  MLPClassifier().fit(X_train, y_train)
prediction = clf.predict(X_test)
accuracy_scores[0] = accuracy_score(y_test, prediction)*100
cm = metrics.confusion_matrix(y_test, prediction)
plt.figure(figsize=(8,8))
plt.grid(b=False)
plot_confusion_matrix(cm, classes=target_names, normalize=True, title='Normalized confusion matrix', cmap = cm_cmap)
report = metrics.classification_report(y_test, prediction, target_names=target_names)
#df = pd.DataFrame(report).transpose()
#df.to_csv('Classification_Report_MLP_Sreeni_Dataset.csv')
print(' MLPClassifier Accuracy: {}%'.format(accuracy_scores[0]))
#cm.numpy.genfromtxt('C:/localpath/test.csv', delimiter=',')
#plt.savefig('Confusion_Matrix_MLP_Participant_Hand_and_Palm_Dataset',transparent=True, bbox_inches='tight', pad_inches=0)
print(report)

gnb = GaussianNB()
clf =  gnb.fit(X_train, y_train)
prediction = clf.predict(X_test)
accuracy_scores[0] = accuracy_score(y_test, prediction)*100
cm = metrics.confusion_matrix(y_test,prediction)
plt.figure(figsize=(8,8))
plt.grid(b=False)
plot_confusion_matrix(cm, classes=target_names, normalize=True, title='Normalized confusion matrix', cmap = cm_cmap)
report = metrics.classification_report(y_test, prediction, target_names=target_names)
#df = pd.DataFrame(report).transpose()
#df.to_csv('Classification_Report_NB_Sreeni_Dataset.csv')
print(' Naive Bayes Classifier Accuracy : {}%'.format(accuracy_scores[0]))
#plt.savefig('Confusion_Matrix_Naive_Bayes_Participant_Hand_and_Palm_Dataset',transparent=True, bbox_inches='tight', pad_inches=0)
print(report)


clf = RandomForestClassifier().fit(X_train, y_train)
prediction = clf.predict(X_test)
accuracy_scores[0] = accuracy_score(y_test, prediction)*100
cm = metrics.confusion_matrix(y_test, prediction)
plt.figure(figsize=(8,8))
plt.grid(b=False)
plot_confusion_matrix(cm, classes=target_names, normalize=True, title='Normalized confusion matrix', cmap = cm_cmap)
report = metrics.classification_report(y_test, prediction, target_names=target_names)
#df = pd.DataFrame(report).transpose()
#df.to_csv('Classification_Report_RF_Sreeni_Dataset.csv')
print('Random Forest Classifier Accuracy: {}%'.format(accuracy_scores[0]))
#plt.savefig('Confusion_Matrix_RF_Participant_Hand_and_Palm_Dataset',transparent=True, bbox_inches='tight', pad_inches=0)
print(report)


clf =  tree.DecisionTreeClassifier().fit(X_train, y_train)
prediction = clf.predict(X_test)
accuracy_scores[0] = accuracy_score(y_test, prediction)*100
cm = metrics.confusion_matrix(y_test, prediction)
plt.figure(figsize=(8,8))
plt.grid(b=False)
plot_confusion_matrix(cm, classes=target_names, normalize=True, title='Normalized confusion matrix', cmap = cm_cmap)
report = metrics.classification_report(y_test, prediction, target_names=target_names)
#df = pd.DataFrame(report).transpose()
#df.to_csv('Classification_Report_DT_Sreeni_Dataset.csv')
print(' Decision Tree Classifier Accuracy: {}%'.format(accuracy_scores[0]))
#plt.savefig('Confusion_Matrix_DT_Participant_Hand_and_Palm_Dataset',transparent=True, bbox_inches='tight', pad_inches=0)
print(report)

clf = SVC(kernel = 'linear').fit(X_train, y_train)
#clf = SVC(kernel='rbf', random_state=0, gamma=1, C=1).fit(X_train, y_train) 
prediction = clf.predict(X_test) 
accuracy_scores[0] = accuracy_score(y_test, prediction)*100
cm = metrics.confusion_matrix(y_test, prediction)
plt.figure(figsize=(8,8))
plt.grid(b=False)
plot_confusion_matrix(cm, classes=target_names, normalize=True, title='Normalized confusion matrix', cmap = cm_cmap)
report = metrics.classification_report(y_test, prediction, target_names=target_names)
#df = pd.DataFrame(report).transpose()
#df.to_csv('Classification_Report_MLP_Sreeni_Dataset.csv')
print('  SVM MulitclassClassifier Accuracy: {}%'.format(accuracy_scores[0]))
#cm.numpy.genfromtxt('C:/localpath/test.csv', delimiter=',')
#plt.savefig('Confusion_Matrix_SVM_Linear_Participant_Hand_and_Palm_Dataset',transparent=True, bbox_inches='tight', pad_inches=0)
print(report)

clf = SVC(kernel = 'rbf').fit(X_train, y_train)
#clf = SVC(kernel='rbf', random_state=0, gamma=1, C=1).fit(X_train, y_train) 
prediction = clf.predict(X_test) 
accuracy_scores[0] = accuracy_score(y_test, prediction)*100
cm = metrics.confusion_matrix(y_test, prediction)
plt.figure(figsize=(8,8))
plt.grid(b=False)
plot_confusion_matrix(cm, classes=target_names, normalize=True, title='Normalized confusion matrix', cmap = cm_cmap)
report = metrics.classification_report(y_test, prediction, target_names=target_names)
#df = pd.DataFrame(report).transpose()
#df.to_csv('Classification_Report_MLP_Sreeni_Dataset.csv')
print('  SVM RBF MulitclassClassifier Accuracy: {}%'.format(accuracy_scores[0]))
#cm.numpy.genfromtxt('C:/localpath/test.csv', delimiter=',')
#plt.savefig('Confusion_Matrix_SVM_RBF_Participant_Hand_and_Palm_Dataset',transparent=True, bbox_inches='tight', pad_inches=0)
print(report)


# In[ ]:


time_start = time.time()
tsne = TSNE(n_components=2, verbose=1, perplexity=10, n_iter=250)
X_tsne = tsne.fit_transform(X)
print('t-SNE done! Time elapsed: {} seconds'.format(time.time()-time_start))


# In[ ]:


time_start = time.time()
tsne = TSNE(n_components=2, verbose=1, perplexity=10, n_iter=250)
x_tsne = tsne.fit_transform(x)
print('t-SNE done! Time elapsed: {} seconds'.format(time.time()-time_start))


# In[ ]:


sns.set(rc={'figure.figsize':(11.7,8.27)})
palette = sns.color_palette("bright", 9)
sns.scatterplot(x=x_tsne[:,0], y=x_tsne[:,1],
            hue = Y,
            alpha=0.9,
            #legend='full',
            palette=palette
                )


# In[ ]:





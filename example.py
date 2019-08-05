from sacred import Experiment
from sacred.observers import MongoObserver
from numpy.random import permutation
from sklearn import svm, datasets

ex = Experiment('iris_rbf_svm')

#MongoDB settings

DATABASE_URL = "172.18.65.219:27017"
DATABASE_NAME = "demo_db"

m_observer = MongoObserver.create(url=DATABASE_URL, db_name=DATABASE_NAME)
ex.observers.append(m_observer)


#hyperparameters
@ex.config
def my_config():
    C = 1.0
    gamma = 0.7


@ex.automain
def run(C, gamma, _run):
        #load a dataset
        iris = datasets.load_iris()
        
        #permute the dataset
        per = permutation(iris.target.size)
        iris.data = iris.data[per]
        iris.target = iris.target[per]
        
        #create an SVM classifier with 'rbf' kernel
        clf = svm.SVC(C, 'rbf', gamma=gamma)
        
        #build train and dev set
        train_x, train_y = iris.data[:90], iris.target[:90]
        dev_x, dev_y = iris.data[90:], iris.target[90:]

        #train classifier
        clf.fit(train_x, train_y)
        
        #get the score
        score = clf.score(dev_x, dev_y)
        
        #save the score
        _run.log_scalar('score', score)

import numpy as np

from sklearn.model_selection import train_test_split
from sklearn.datasets import make_moons
from sklearn.ensemble import BaggingClassifier
from sklearn.tree import DecisionTreeClassifier

if __name__=='__main__':

    import sys, os

    # visualization module
    from datavyz import ge

    X, y = make_moons(n_samples=400, noise=0.30, random_state=42)
    X_train, X_test, y_train, y_test = train_test_split(X, y, random_state=42)

    
    # Single Decision Tree Classifier
    tree = DecisionTreeClassifier()
    tree.fit(X_train, y_train)

    # Bagging Classifier
    bag_clf = BaggingClassifier(\
        DecisionTreeClassifier(),
        n_estimators=500, max_samples=0.5, bootstrap=True)
    bag_clf.fit(X_train, y_train)
    

    fig, AX = ge.figure(axes=(1,2))
    for ax in AX:
        ge.scatter(X=[X_train[:,0][y_train==1],X_train[:,0][y_train==0]],
               Y=[X_train[:,1][y_train==1],X_train[:,1][y_train==0]],
               xlabel='x1', ylabel='x2', COLORS=[ge.blue, ge.orange],
               LABELS=['y=1', 'y=0'], ax=ax)


    x1, x2 = np.meshgrid(np.linspace(X_train[:,0].min(), X_train[:,0].max(), 200),
                         np.linspace(X_train[:,1].min(), X_train[:,1].max(), 200))
    y_pred_full = tree.predict(np.array([x1.flatten(), x2.flatten()]).T)
    ge.twoD_plot(x1.flatten(), x2.flatten(), y_pred_full, alpha=0.3, ax=AX[0])

    y_pred_full = bag_clf.predict(np.array([x1.flatten(), x2.flatten()]).T)
    ge.twoD_plot(x1.flatten(), x2.flatten(), y_pred_full, alpha=0.3, ax=AX[1])

    # print(tree.)
    ge.show()

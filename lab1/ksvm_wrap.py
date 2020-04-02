import numpy as np
import matplotlib.pyplot as plt
import data
from sklearn.svm import SVC

class KSVMWrap():
    def __init__(self, X, Y_, param_svm_c=1, param_svm_gamma='auto'):
        # Konstruira omotač i uči RBF SVM klasifikator
        # X, Y_:           podatci i točni indeksi razreda
        # param_svm_c:     relativni značaj podatkovne cijene
        # param_svm_gamma: širina RBF jezgre

        self.clf = SVC(C=param_svm_c, gamma=param_svm_gamma, probability=True)
        self.clf.fit(X, Y_)

    def predict(self, X):
        # Predviđa i vraća indekse razreda podataka X
        return self.clf.predict(X)

    def get_scores(self, X):
        # Vraća klasifikacijske mjere
        # (engl. classification scores) podataka X;
        # ovo će vam trebati za računanje prosječne preciznosti.
        return self.clf.predict_proba(X)

    def support(self):
        # Indeksi podataka koji su odabrani za potporne vektore
        return self.clf.support_


def svm_decfun(model):
    return lambda X: model.get_scores(X)[np.arange(len(X)), 1]


if __name__ == "__main__":
    # inicijaliziraj generatore slučajnih brojeva
    np.random.seed(100)

    # instanciraj podatke X i labele Yoh_
    X, Y_ = data.sample_gmm_2d(6, 2, 10)

    # definiraj model:
    svm = KSVMWrap(X, Y_, 10, 'auto')

    # dohvati vjerojatnosti na skupu za učenje
    Y = svm.predict(X)

    # ispiši performansu (preciznost i odziv po razredima)
    accuracy, pr, _ = data.eval_perf_multi(Y, Y_)
    print(f'accuracy: {accuracy}, precision: {pr[0]}, recall: {pr[1]}')

    # iscrtaj rezultate, decizijsku plohu
    decfun = svm_decfun(svm)
    bbox=(np.min(X, axis=0), np.max(X, axis=0))
    data.graph_surface(decfun, bbox, offset=0.5)
    data.graph_data(X, Y_, Y, special=svm.support())

    # Prikaži
    plt.show()
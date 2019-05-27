from sklearn.linear_model import LogisticRegression, LogisticRegressionCV
from sklearn.model_selection import cross_val_score
from sklearn.decomposition import PCA

class Classifier():
    def __init__(self, data, target):
        self.data = data
        self.target = target

    def _predict(self):
        solvers_l2 = ['newton-cg', 'lbfgs', 'sag', 'saga', 'liblinear']
        results_l2 = {}
        for solver in solvers_l2:
            clf = LogisticRegression(solver=solver, penalty='l2', multi_class='auto')
            cv3 = cross_val_score(clf, self.data, self.target, cv=3)
            cv5 = cross_val_score(clf, self.data, self.target, cv=5)
            v3 = sum(cv3) / len(cv3)
            v5 = sum(cv5) / len(cv5)
            results_l2[solver] = (v3, 3) if v3 > v5 else (v5, 5)

        solvers_l1 = ['saga', 'liblinear']
        results_l1 = {}
        for solver in solvers_l1:
            clf = LogisticRegression(solver=solver, penalty='l1', multi_class='auto')
            cv3 = cross_val_score(clf, self.data, self.target, cv=3)
            cv5 = cross_val_score(clf, self.data, self.target, cv=5)
            v3 = sum(cv3) / len(cv3)
            v5 = sum(cv5) / len(cv5)
            results_l1[solver] = (v3, 3) if v3 > v5 else (v5, 5)

        max_l2 = max(results_l2, key=results_l2.get)
        max_l1 = max(results_l1, key=results_l1.get)
        if results_l1[max_l1][0] > results_l2[max_l2][0]:
            best_solver = (max_l1, 'l1', results_l1[max_l1][1], results_l1[max_l1][0])
        else:
            best_solver = (max_l2, 'l2', results_l2[max_l2][1], results_l2[max_l2][0])

        return best_solver

    def _pca(self):
        data = self.data
        solvers = []
        for n in range(1,6):
            pca = PCA(n_components=n).fit_transform(data, self.target)
            self.data = pca
            solvers.append(self._predict())
        max_score = 0
        max_idx = 0
        for idx, solver in enumerate(solvers):
            if solver[3] > max_score:
                max_score = solver[3]
                max_idx = idx

        self.data = data
        return idx, solvers[max_idx]

    def classify(self, data_test, data_target, solver, pca_n=None):
        data = self.data
        if pca_n:
            data = PCA(n_components=pca_n).fit_transform(data, self.target)
            data_test = PCA(n_components=pca_n).fit_transform(data_test, data_target)

        clf = LogisticRegressionCV(solver=solver[0], penalty=solver[1], cv=solver[2], multi_class='auto').fit(data, self.target)
        return clf.predict(data_test)


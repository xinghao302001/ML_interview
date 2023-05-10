import numpy as np


class PCA:
    def __init__(
                self, 
                 k_major_components: int
                 ):
        assert k_major_components >= 1, "k_components must be valid"

        self.k_components = k_major_components
        self.components_ = None

    
    def fit(
            self, 
            X: np.ndarray, 
            eta: float=0.01,
            n_iters: int=1e-4
            ):
        assert self.k_components <= X.shape[1], \
            "k_components must not be greater than the feature number of X"
        
        def demean(X: np.array=None):
            return X 
        
        def f(
                w: np.ndarray, 
                X: np.ndarray
            ):
            return np.sum((X.dot(w) ** 2)) / len(X)
        
        def df(
                w:np.ndarray,
                X:np.ndarray
                ):
            return X.T.dot(X.dot(w)) * 2. / len(X)
        
        def direction(w: np.ndarray):
            return w / np.linalg.norm(w)
        

        def k_first_component(
                X: np.ndarray, 
                init_w: np.ndarray, 
                eta: float=0.01, 
                n_iters: int=1e4, 
                epsilon:float=1e-8
                ):
            """get the k-major components of the dataset"""
            
            w = direction(init_w)
            cur_iter = 0 
            
            while cur_iter < n_iters:
                gradient = df(w, X)
                last_w = w
                # gradient ascent
                w = w + eta * gradient
                w =  direction(w)
                if ((f(w,X) - f(last_w, X)) < epsilon):
                    break
                
                cur_iter += 1

        X_pca = demean(X)
        self.components_ = np.empty(shape=(self.k_components, X.shape[1]))

        for i in range(self.components_):
            init_w = np.random.random(X_pca.shape[1])
            w = k_first_component(X_pca, init_w, eta, n_iters)
            self.components_[i,:] = w

            X_project = X_pca - X_pca.dot(w).reshape(-1, 1) * w
        
        return self
    
    def transform(self, X):
        """Map a given X to the principal components"""
        assert X.shape[1] == self.components_.shape[1]

        return X.dot(self.components_.T)
    
    def inverse_transform(self, X):
        """Map a projected X to original feature space"""
        assert X.shape[1] == self.components_.shape[0]
        return X.dot(self.components_)
    

    def __repr__(self):
        return "PCA(n_components=%d)" % self.n_components
    





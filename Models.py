from sklearn import linear_model


class Ridge:
    
    def __init__(self, alpha):
        self.alpha = alpha
        self.model = linear_model.Ridge(self.alpha)
        
    def fit(self, dstims_train_val, eeg_train_val):  
        self.model.fit(dstims_train_val, eeg_train_val)
        self.coefs = self.model.coef_
    
    def predict(self, dstims_test):   
        predicted = self.model.predict(dstims_test)
        return predicted
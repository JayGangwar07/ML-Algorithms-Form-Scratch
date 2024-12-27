class DecisionTree:
    def __init__(self,min_split=2,max_depth=5):
        self.min_split = 2
        self.max_depth = 5
        self.tree = None

    def fit(self,x,y):
        self.tree = self.build_tree(x,y)

#Helper functions

    def build_tree(self,x,y,depth = 0):
        n_samples,num_features = x.shape
        if n_samples >= self.min_split and depth <= self.max_depth:
            best_split = self.find_split(x,y,num_features)
            if best_split["gain"]>0:
                left_tree = self.build_tree(best_split["x_left"],best_split["y_left"],depth+1)
                right_tree = self.build_tree(best_split["x_right"],best_split["y_right"],depth+1)
                return {"feature_index": best_split["feature_index"],
                    "threshold": best_split["threshold"],
                    "left": left_tree,
                    "right": right_tree}
        return {"leaf":True,"value":self.leaf_value(y)}

    def find_split(self,x,y,num_features):
        best_split = {"gain":-1}
        for i in range(num_features):
            feature_values = x[:,i]
            threshold = np.unique(feature_values)
            for j in threshold:
                x_left, y_left, x_right, y_right = self._split(x,y,i,j)#j
                if len(y_left)>0 and len(y_right)>0:
                    gain = self.info_gain(y,y_left,y_right)
                    if gain>best_split["gain"]:
                        best_split = {"feature_index": i, "threshold": j,
                                  "x_left": x_left, "y_left": y_left,
                                  "x_right": x_right, "y_right": y_right,
                                  "gain": gain}
        return best_split

    def _split(self,x,y,i,j):
        left_mask = x[:,i] <= j
        right_mask = ~left_mask
        return x[left_mask],y[left_mask],x[right_mask], y[right_mask]

    def info_gain(self,y,y_left,y_right):
        p = len(y_left) / len(y)
        return self.gini(y)-(p*self.gini(y_left)+(1-p)*self.gini(y_right))

    def gini(self,y):
        m = len(y)
        if m == 0:
           return 0
        p = np.bincount(y)
        return 1-np.sum((p/m)**2)

    def leaf_value(self,y):
        return np.bincount(y).argmax()

#Main predict function

    def predict(self,x):
        predictions = [self.predict_sample(sample,self.tree) for sample in x]
        return np.array(predictions)

#Helper function

    def predict_sample(self,sample,tree):
        if "leaf" in tree:
            return tree["value"]
        feature_value = sample[tree["feature_index"]]
        if feature_value <= tree["threshold"]:
            return self.predict_sample(sample,tree["left"])
        else:
            return self.predict_sample(sample,tree["right"])

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import time

'''---------------------------------------------- LINEAR REGRESSION -------------------------------------------------'''
class LinearRegreassion:
    def __init__(self, num_epochs=1, loss_improvement=1e-6, epsilon=1e-3, satisfied_stability=1e3, momentum=0, lr=1e-1, lr_decay=0, use_miniBatch=False, with_bias=True):
        self.bias = None
        self.num_epochs = num_epochs
        self.loss_improvement = loss_improvement
        self.epsilon = epsilon
        self.satisfied_stability = satisfied_stability
        self.momentum = momentum
        self.lr = lr
        self.lr_decay = lr_decay # coefficient that controls the rate at which the lr decreases
        self.use_miniBatch = use_miniBatch
        self.with_bias = with_bias
        self.grad_norm_hist = []
        self.time_hist = []

    def fit(self, X, Y, batch_size=None):
        self.t = time.time() # training start time
        if self.with_bias: 
            X = self.add_bias(X)
        D_features = X.shape[1] # D_features: the number of features
        # Initialize weights
        self.w = np.zeros(D_features)
        # save original data for later use
        X_origin = X
        Y_origin = Y

        if self.use_miniBatch==True: # use miniBatchSGD, otherwise use analytical linear regression
            gradient = np.full(self.w.shape,np.inf)
            self.w = np.zeros(D_features)
            delta_w = np.zeros(D_features)
            loss_stability = 0
            loop_counter = 0
            prev_loss = float('inf')
            for epoch in range(self.num_epochs):
                epoch_loss = 0  # Initialize the loss for the epoch
                mini_batches = get_mini_batch(X, Y, batch_size)
                for x_mini_batch, y_mini_batch in mini_batches:
                    gradient = self.gradient(x_mini_batch, y_mini_batch, self.w)
                    delta_w = self.momentum*delta_w + (1.0-self.momentum) * gradient
                    self.w = self.w - self.lr*delta_w # update weights
                    self.lr = self.lr / (1.0 + self.lr_decay*loop_counter) # update lr based on number of iterations
                    loop_counter += 1
  
                    # #check if training should be terminated: Loss Improvement-Based Termination    
                    loss = self.loss_function(x_mini_batch, y_mini_batch)
                    grad_norm = np.linalg.norm(gradient)
   
                    if np.abs(prev_loss - loss) < self.loss_improvement:
                        loss_stability += 1
                        if loss_stability > self.satisfied_stability and loss < self.epsilon:
                            break
                    else:
                        loss_stability = 0
                    prev_loss = loss
                if loss_stability > self.satisfied_stability:
                    break
                self.grad_norm_hist.append(grad_norm)
            self.t = time.time()-self.t # calculate time taken for training
            self.time_hist.append(self.t)
            # Report training statistics
            print(f'Mini-batch SGD completed in {round(self.t, 3)} seconds.')
            print(f'Iterations: {loop_counter}, Batch size: {batch_size}')            
        # use analytical linear regression
        else:
            self.compute_w(X_origin, Y_origin)
            self.t = time.time()-self.t # calculate time taken for training
            print(f'Analytical solution computed in {round(self.t, 3)} seconds.')

    def add_bias(self, X):
        N_samples = X.shape[0]
        self.bias = np.ones((N_samples, 1))
        X_with_bias = np.append(self.bias, X, axis=1)
        return X_with_bias
    
    def predict(self, X):
        y_pred = np.dot(X, self.w)
        return y_pred

    def loss_function(self, X, y_true):
        y_pred = self.predict(X)
        residual = y_pred - y_true
        MSE = np.mean(residual**2)
        return MSE
    
    def compute_w(self, X_origin, Y_origin):
        '''analytical linear regression solution: use closed form solution'''
        xtx_inverse = np.linalg.inv(np.matmul(X_origin.T, X_origin))
        xtx_inverse_xt = np.matmul(xtx_inverse, X_origin.T)
        self.w = np.matmul(xtx_inverse_xt, Y_origin)

    def gradient(self, x, y, w):
        N_samples = x.shape[0]  # number of samples
        xw = np.dot(x, w)
        gradient_jw = np.dot(x.T, xw-y)/N_samples
        return gradient_jw
 
'''---------------------------------------------- LOGISTIC REGRESSION -------------------------------------------------'''
class LogisticRegression:
    def __init__(self, boundary=0.5, num_epochs=1, loss_improvement=1e-6, epsilon=1e-3, satisfied_stability=1e3, lr=1e-2, lr_decay=0, momentum=0, use_miniBatch=False, with_bias=True):
        self.boundary=boundary
        self.num_epochs = num_epochs
        self.loss_improvement = loss_improvement
        self.epsilon = epsilon
        self.satisfied_stability = satisfied_stability
        self.lr = lr
        self.lr_decay = lr_decay
        self.momentum = momentum
        self.use_miniBatch = use_miniBatch
        self.with_bias = with_bias
        self.grad_norm_hist = []
        self.time_hist = []

    def fit(self, x, y, batch_size=None):
        self.t = time.time() # training start time
        if self.with_bias: 
            x = self.add_bias(x)
        N,D = x.shape 
        # num_classes = len(np.unique(y))
        self.w = np.random.rand(D) * 0.01  # Small random values
        gradient = np.full(self.w.shape,np.inf)
        delta_w = np.zeros(D)
        loss_stability = 0
        loop_counter = 0
        prev_loss = float('inf')

        model = "miniBatchSGD"
        if self.use_miniBatch == False:
            batch_size = N # Use full batch if mini-batch is not specified
            model = "full batch"
        for epoch in range(self.num_epochs):
            epoch_loss = 0  # Initialize the loss for the epoch
            mini_batches = get_mini_batch(x, y, batch_size)
            for x_mini_batch, y_mini_batch in mini_batches:
                gradient = self.gradient(x_mini_batch, y_mini_batch, self.w)
                delta_w = self.momentum*delta_w + (1.0-self.momentum) * gradient
                self.w = self.w - self.lr*delta_w # update weights
                self.lr = self.lr / (1.0 + self.lr_decay*loop_counter) # update lr based on number of iterations
                loop_counter += 1

                # #check if training should be terminated: Loss Improvement-Based Termination    
                loss = self.loss_function(x_mini_batch, y_mini_batch)
                grad_norm = np.linalg.norm(gradient)

                if np.abs(prev_loss - loss) < self.loss_improvement:
                    loss_stability += 1
                    if loss_stability > self.satisfied_stability and loss < self.epsilon:
                        break
                else:
                    loss_stability = 0
                prev_loss = loss
            if loss_stability > self.satisfied_stability:
                break
            self.grad_norm_hist.append(grad_norm)
        self.t = time.time()-self.t # calculate time taken for training
        self.time_hist.append(self.t)
        # Report training statistics
        print(f'{model} completed in {round(self.t, 3)} seconds.')
        if self.use_miniBatch == True: 
            print(f'Iterations: {loop_counter}, Batch size: {batch_size}')
        else: 
            print(f'Iterations: {loop_counter}')  
            
    def sigmoid(self, z):
        sigmoid = 1 / (1 + np.exp(-z))
        return sigmoid

    def predict(self, x):
        z = np.dot(x, self.w) 
        y_pred = self.sigmoid(z) # apply squshing function to the linear function
        return (y_pred >= self.boundary).astype(int)
    
    def loss_function(self, x, y):
        N, D = x.shape                                                       
        z = np.dot(x, self.w)
        J = np.mean(y * np.log1p(np.exp(-z)) + (1-y) * np.log1p(np.exp(z)))  #log1p calculates log(1+x) to remove floating point inaccuracies 
        return J

    def gradient(self, x, y, w):
        N, D = x.shape
        xw = np.dot(x, w)
        sig_xw = self.sigmoid(xw)
        gradient_jw = np.dot(x.T, sig_xw-y)/N
        return gradient_jw   
    
    def add_bias(self, X):
        N_samples = X.shape[0]
        self.bias = np.ones((N_samples, 1))  # Create a column of ones
        X_with_bias = np.concatenate((self.bias, X), axis=1) 
        return X_with_bias

'''------------------------------------------------- HELPER FUNCTION --------------------------------------------------'''
def split_train_test_data(X_scaled, y, train_ratio, random_state):
    np.random.seed(random_state)  # Set seed for reproducibility
    total_rows = X_scaled.shape[0]
    
    indices = np.arange(total_rows)
    np.random.shuffle(indices)

    split_index = int(train_ratio * total_rows) 

    # Use regular indexing for NumPy arrays
    X_train, X_test = X_scaled[indices[:split_index]], X_scaled[indices[split_index:]]
    y_train, y_test = y[indices[:split_index]], y[indices[split_index:]]

    return X_train, X_test, y_train, y_test

def get_mini_batch(X, Y, batch_size):
    N_samples = X.shape[0]
    indices = np.arange(N_samples)

    # Shuffle the indices to ensure randomness in the mini-batches
    np.random.shuffle(indices) 
    X_shuffled = X[indices]
    Y_shuffled = Y[indices]

    # Create mini-batches
    mini_batches = []
    for i in range(0, N_samples, batch_size):
        X_mini_batch = X_shuffled[i:i + batch_size]
        Y_mini_batch = Y_shuffled[i:i + batch_size]
        mini_batches.append((X_mini_batch, Y_mini_batch))

    return mini_batches

def generate_synthetic_samples(X, y):
    y_df = pd.DataFrame(y, columns=['target'])
    minority_class = y_df['target'].value_counts().idxmin() # find minority class
    
    # Get indices of minority class samples
    minority_indices = y_df[y_df['target'] == minority_class].index
    X_minority = X[minority_indices]
    
    # Generate synthetic samples
    synthetic_samples = []
    total_samples = len(X_minority)
    for i in range(total_samples):
        # Randomly choose another sample from the minority class
        random_index = np.random.choice(minority_indices)
        synthetic_sample = (X_minority[i % len(X_minority)] + X[random_index]) / 2  # Simple averaging
        synthetic_samples.append(synthetic_sample)
    
    # Append synthetic samples to the dataset
    X_balanced = np.vstack([X, np.array(synthetic_samples)])
    y_balanced = np.concatenate([y, np.array([minority_class] * total_samples)])  # Reshape to 2D

    return X_balanced, y_balanced

def r2_score(y_true, y_pred):
    """ goodness of fit of a regression model """
    RSS = np.sum((y_true - y_pred) ** 2) # sum of squares of residuals
    TSS = np.sum((y_true - np.mean(y_true)) ** 2) # total sum of squares
    r2 = 1 - (RSS / TSS)
    return r2

def mean_absolute_error(y_true, y_pred):
    """ average of the absolute errors between the predicted and true values """
    MAE = np.mean(np.abs(y_true - y_pred))
    return MAE

def accuracy(y_true, y_pred):
    """ proportion of correct predictions out of the total predictions """
    return np.sum(y_true==y_pred) / len(y_true)

def calculate_precision(y_true, y_pred, class_label):
    """ proportion of true positive predictions out of all the positive predictions """
    precisions = []
    for label in class_label:
        TP = np.sum((y_true == label) & (y_pred == label)) # True Positives (TP) for class_label
        PP = np.sum(y_pred == label) # Predictied Positives (PP) for class_label
        if PP == 0:
            precision = 0  # To avoid division by zero
        else: 
            precision =  TP / PP
        precisions.append(precision)
    return precisions

def calculate_recall(y_true, y_pred, class_label):
    recalls = []
    for label in class_label:
        TP = np.sum((y_true == label) & (y_pred == label)) 
        FN = np.sum((y_true == label) & (y_pred != label)) # False Negatives (FN) for class_label
        recall = TP / (TP + FN) if (TP + FN) != 0 else 0
        recalls.append(recall) 
    return recalls 

def calculate_f1_score(y_true, y_pred, class_label):
    """ harmonic mean of precision and recall. useful when dataset is imbalanced """
    f1_scores = []
    precisions = calculate_precision(y_true, y_pred, class_label)
    recalls = calculate_recall(y_true, y_pred, class_label)
    for prec, rec in zip(precisions, recalls):
        if prec + rec == 0:
            f1 = 0
        else:
            f1 = 2 * (prec * rec) / (prec + rec)
        f1_scores.append(f1)
    return f1_scores

def performance_log(y_train_log, y_train_pred, y_test_log, y_test_pred, class_label):
    train_accuracy = accuracy(y_train_log, y_train_pred) # train set
    train_precision = calculate_precision(y_train_log, y_train_pred, class_label) # train set
    train_recall = calculate_recall(y_train_log, y_train_pred, class_label) # train set
    train_f1 = calculate_f1_score(y_train_log, y_train_pred, class_label) # train set

    test_accuracy = accuracy(y_test_log, y_test_pred) # test set
    test_precision = calculate_precision(y_test_log, y_test_pred, class_label) # test set
    test_recall = calculate_recall(y_test_log, y_test_pred, class_label) # test set
    test_f1 = calculate_f1_score(y_test_log, y_test_pred, class_label) # test set
    print(f"Train set:\n"
        f"accuracy: {train_accuracy:.4f}, "
        f"precision: {[f'{p:.4f}' for p in train_precision]}, "
        f"recall: {[f'{r:.4f}' for r in train_recall]}, "
        f"f1-score: {[f'{f1:.4f}' for f1 in train_f1]}, "
        f"\nTest set:\n"
        f"accuracy: {test_accuracy:.4f}, "
        f"precision: {[f'{p:.4f}' for p in test_precision]}, "
        f"recall: {[f'{r:.4f}' for r in test_recall]}, "
        f"f1-score: {[f'{f1:.4f}' for f1 in test_f1]}")
    return train_accuracy, train_precision, train_recall, train_f1, test_accuracy, test_precision, test_recall, test_f1

def performance_linear(y_train, y_test, y_train_pred, y_test_pred):
    train_R2 = r2_score(y_train, y_train_pred) # train set
    train_MAE = mean_absolute_error(y_train, y_train_pred) # train set
    test_R2 = r2_score(y_test, y_test_pred) # test set
    test_MAE = mean_absolute_error(y_test, y_test_pred) # test set
    print(f"Train set:\nMAE: {train_MAE:.4f}, R2: {train_R2:.4f} \nTest set:\nMAE: {test_MAE:.4f}, R2: {test_R2:.4f}")
    return train_R2, train_MAE, test_R2, test_MAE

def w_plot(feature_names, w, title):
    if feature_names.shape != w.shape:
        w = w[:-1]
    plt.figure(figsize=(12, 8)) 
    plt.barh(feature_names, w, color='skyblue')
    plt.xlabel('Weight Value')
    plt.ylabel('Features')
    plt.title(title)
    plt.yticks(rotation=0)
    plt.grid(axis='x', linestyle='--', alpha=0.7)
    plt.show()

def size_vs_performance(X, y, sizes, model, batch_sizes, class_label=None, linear=True):
    train_MSEs = []
    test_MSEs = []
    train_R2s = []
    train_MAEs = []
    test_R2s = []
    test_MAEs = []
    train_accuracys =[] 
    train_precisions = []
    train_recalls = []
    train_f1s = []
    test_accuracys = []
    test_precisions = [] 
    test_recalls = [] 
    test_f1s = []
    grad_norm_hists = []
    for size in sizes:
        x_train, x_test, y_train, y_test = split_train_test_data(X, y, train_ratio=size/100, random_state=42)
        # add bias term
        for batch_size in batch_sizes:
            model.fit(x_train, y_train, batch_size=batch_size)
            grad_norm_hists.append(model.grad_norm_hist) 
            if model.with_bias:
                x_test = model.add_bias(x_test)
                x_train = model.add_bias(x_train)
            y_train_pred = model.predict(x_train)
            y_test_pred = model.predict(x_test)

            train_MSE = model.loss_function(x_train, y_train)
            test_MSE = model.loss_function(x_test, y_test)
            if linear:
                train_R2, train_MAE, test_R2, test_MAE = performance_linear(y_train, y_test, y_train_pred, y_test_pred)
                train_MSEs.append(train_MSE)
                train_R2s.append(train_R2)
                train_MAEs.append(train_MAE)
                test_MSEs.append(test_MSE)
                test_R2s.append(test_R2)
                test_MAEs.append(test_MAE)
            else:
                train_accuracy, train_precision, train_recall, train_f1, test_accuracy, test_precision, test_recall, test_f1 = performance_log(y_train, y_train_pred, y_test, y_test_pred, class_label)
                train_MSEs.append(train_MSE)
                test_MSEs.append(test_MSE)
                train_accuracys.append(train_accuracy) 
                train_precisions.append(train_precision) 
                train_recalls.append(train_recall) 
                train_f1s.append(train_f1) 
                test_accuracys.append(test_accuracy) 
                test_precisions.append(test_precision) 
                test_recalls.append(test_recall)  
                test_f1s.append(test_f1)
    if linear:
        return train_MSEs, test_MSEs, train_R2s, train_MAEs, test_R2s, test_MAEs, grad_norm_hists
    else:
        return train_MSEs, train_accuracys, train_precisions, train_recalls, train_f1s, test_MSEs, test_accuracys, test_precisions, test_recalls, test_f1s, grad_norm_hists

def size_plot(size, train_perform, test_perform, y_label, x_label):
    plt.figure(figsize=(10, 6))
    label_train = "train set " + y_label
    label_test = "test set " + y_label
    plt.plot(size, train_perform, label=label_train, marker='o', color='blue')
    plt.plot(size, test_perform, label=label_test, marker='x', color='red')
    plt.title(f'{y_label} vs {x_label}')
    plt.xlabel(f'{x_label}')
    plt.ylabel(f'{y_label}')
    plt.legend()
    plt.grid(True)
    plt.show()

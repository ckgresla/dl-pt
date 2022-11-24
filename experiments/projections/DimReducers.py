#!/home/ckg/miniconda3/envs/pt/bin/python
# Implementations of PCA & T-SNE


import numpy as np
np.seterr(all='raise') #handle nan/inf error (in eigenvec computation)




# PCA Class (referenced– https://medium.com/accel-ai/pca-algorithm-tutorial-in-python-93ff19212026)
class PCA:


    def run(self, data, n_components=2):
        #Components=2 is nice default for visualization
        self.n_samples = data.shape[0] #assuming num of instances is on first axis
        self.n_components = n_components

        # Normalize Data (1st step in PCA, center values around mean with unit standard deviation)
        self.standardized_data = self.standardization(data)

        # Compute CoVariance Matrix & Eigenvectors
        covariance_matrix = self.cm() #references the standardized_data under the hood
        eigenvectors = self.eigenvectors(covariance_matrix)

        # Project Standardized Data w Eigenvectors
        projected_data = self.projection(eigenvectors)

        return projected_data


    def standardization(self, data):
        # Scale Data for Unit Std Dev centered around Mean
        data = data + 1e-12 #add epsilon to avoid divide by zero
        z = (data - np.mean(data, axis=0)) / (np.std(data, axis=0)) #computed at row level?
        return z


    # Covariance Matrix Func
    def cm(self, degrees_freedom=0):
        covariance_matrix = np.dot(self.standardized_data.T, self.standardized_data)/(self.n_samples-degrees_freedom)
        return covariance_matrix


    # Get Eigenvectors (sort for importance, return N best as specified in `n_components`)
    def eigenvectors(self, covariance_matrix):
        eigenvalues, eigenvectors = np.linalg.eig(covariance_matrix)
        n_cols = np.argsort(eigenvalues)[::-1][:self.n_components]
        selected_vecs = eigenvectors[:,n_cols]
        return selected_vecs


    # Dot Product to linearly project
    def projection(self, eigenvecs):
        P = np.dot(self.standardized_data, eigenvecs)
        return P


    # Wrapper to Compute PCA Nicely --> returns best `n_components`
    def compute(self, data_matrix, n_components=2):
        data_matrix = np.array(data_matrix) #convert list of tensors into a numpy ndarray of correct dim: (70k x 784)
        projected_data_matrix = self.run(data_matrix, n_components) #2 for 2d, 3 for 3d (controls dimensions for Principal Components)
        return projected_data_matrix



import numpy as np
import scipy.special
import math
import pickle
import functools
import time
from scipy.stats import truncnorm
from functools import wraps
import time


#Main.py processes data from the MNIST database. It will create and train a neural network
#and process images. 

#1. Download mnist_train.csv and mnist_test.csv from https://www.python-course.eu/neural_network_mnist.php into a directory 
#on your computer. Change the data_path global variable in Main.py to the path of that directory. 
#2. Run processData() in the main() method in Main.py to process the greyscaled images. Raw pixel values range from 
#0 to 255; processData() converts them to values in the range [0.01, 1]. ProcessData() also 
#converts the raw labels (0-10 corresponding to the correct digit for each image) into a one-hot
#representation. For example, a label of 2 is converted to [0,0,1,0,0,0,0,0,0,0]. The processed
#test and train images and labels are pickled and saved at data_path/pickled_mnist.pkl
#3. Call trainNetwork() to calculate and store weights for the neural network. The method will
#default to a 3-layer network with 80 nodes in layers 1 and 2 and 10 output nodes in layer 3, a 
#learning rate of .1 (which determines the rate of gradient descent) and 3 runs through the same
#training data to continually readjust the weights matrix. The final weights matrices are stored in data_path/pickled_weights.pkl.
#TrainNetwork() calls nn.train(), which runs nn.feedforward(x) and nn.backbrop(x,y), using the training data.
#It then evaluates the network's performance via nn.evaluate(), which uses the training data.
#4.Call singleResult(x) to get the network's output for a given input x (an array of length 784 corresponding to an input image).
#SingleResult(x) loads data_path/pickled_weights.pkl into a neural network, then runs nn.feedforward(x), returning
#an array with the one-hot representation of the network's output. 

#Note that after calling processData() you need not call it again. Also, you only need to call trainNetwork()
#again to update weights, for example because you want to change the network structure.



data_path = "/Users/anthonyraubenheimer/Sites/numberclassifier/pickled_data/"
network_structure = [784,80,80,10]
learning_rate = .01
num_runs = 5

def timer(func): #from https://realpython.com/lessons/timing-functions-decorators/
    """Print the runtime of the decorated function"""
    @functools.wraps(func)
    def wrapper_timer(*args, **kwargs):
        start_time = time.perf_counter()  # 1
        value = func(*args, **kwargs)
        end_time = time.perf_counter()  # 2
        run_time = end_time - start_time  # 3
        print(f"Finished {func.__name__!r} in {run_time:.1f} secs")
        return value
    return wrapper_timer

def main():
    nn = NeuralNetwork(network_structure, learning_rate, None)
    nn.processData()
    nn.trainNetwork()
    
    #test input corresponding to an image of a handwritten "2"
    #input = [0.01, 0.01, 0.01, 0.01, 0.01, 0.01, 0.01, 0.01, 0.01, 0.01, 0.01, 0.01, 0.01, 0.01, 0.01, 0.01, 0.01, 0.01, 0.01, 0.01, 0.01, 0.01, 0.01, 0.01, 0.01, 0.01, 0.01, 0.01, 0.01, 0.01, 0.01, 0.01, 0.01, 0.01, 0.01, 0.01, 0.01, 0.01, 0.01, 0.01, 0.01, 0.01, 0.01, 0.01, 0.01, 0.01, 0.01, 0.01, 0.01, 0.01, 0.01, 0.01, 0.01, 0.01, 0.01, 0.01, 0.01, 0.01, 0.01, 0.01, 0.01, 0.01, 0.01, 0.01, 0.01, 0.25070588235294117, 0.25070588235294117, 0.10317647058823529, 0.01, 0.01, 0.01, 0.01, 0.01, 0.01, 0.01, 0.01, 0.01, 0.01, 0.01, 0.01, 0.01, 0.01, 0.01, 0.01, 0.01, 0.01, 0.01, 0.01, 0.01, 0.01, 0.01, 0.3749411764705883, 1, 1, 1, 1, 0.9495294117647058, 0.9844705882352942, 0.01, 0.01, 0.01, 0.01, 0.01, 0.01, 0.01, 0.01, 0.01, 0.01, 0.01, 0.01, 0.01, 0.01, 0.01, 0.01, 0.01, 0.01, 0.01, 0.01, 1, 1, 1, 1, 1, 1, 1, 1, 1, 0.01, 0.01, 0.01, 0.01, 0.01, 0.01, 0.01, 0.01, 0.01, 0.01, 0.01, 0.01, 0.01, 0.01, 0.01, 0.01, 0.01, 0.01, 0.01, 0.9301176470588235, 1, 1, 0.08764705882352941, 0.01, 0.3710588235294118, 1, 1, 1, 1, 0.01, 0.01, 0.01, 0.01, 0.01, 0.01, 0.01, 0.01, 0.01, 0.01, 0.01, 0.01, 0.01, 0.01, 0.01, 0.01, 0.01, 0.01, 0.01, 0.01, 0.01, 0.01, 0.01, 0.01, 0.01, 1, 1, 1, 0.01, 0.01, 0.01, 0.01, 0.01, 0.01, 0.01, 0.01, 0.01, 0.01, 0.01, 0.01, 0.01, 0.01, 0.01, 0.01, 0.01, 0.01, 0.01, 0.01, 0.01, 0.01, 0.01, 0.01, 0.01, 0.01, 1, 1, 0.7049411764705883, 0.01, 0.01, 0.01, 0.01, 0.01, 0.01, 0.01, 0.01, 0.01, 0.01, 0.01, 0.01, 0.01, 0.01, 0.01, 0.01, 0.01, 0.01, 0.01, 0.01, 0.01, 0.01, 0.01, 0.01, 0.01, 1, 1, 1, 0.01, 0.01, 0.01, 0.01, 0.01, 0.01, 0.01, 0.01, 0.01, 0.01, 0.01, 0.01, 0.01, 0.01, 0.01, 0.01, 0.01, 0.01, 0.01, 0.01, 0.01, 0.01, 0.01, 0.01, 0.01, 1, 1, 1, 0.01, 0.01, 0.01, 0.01, 0.01, 0.01, 0.01, 0.01, 0.01, 0.01, 0.01, 0.01, 0.01, 0.01, 0.01, 0.01, 0.01, 0.01, 0.01, 0.01, 0.01, 0.01, 0.01, 0.01, 0.01, 1, 1, 0.5962352941176471, 0.01, 0.01, 0.01, 0.01, 0.01, 0.01, 0.01, 0.01, 0.01, 0.01, 0.01, 0.01, 0.01, 0.01, 0.01, 0.01, 0.01, 0.01, 0.01, 0.01, 0.01, 0.01, 0.01, 0.01, 1, 1, 0.01, 0.01, 0.01, 0.01, 0.01, 0.01, 0.01, 0.01, 0.01, 0.01, 0.01, 0.01, 0.01, 0.01, 0.01, 0.01, 0.01, 0.01, 0.01, 0.01, 0.01, 0.01, 0.01, 0.01, 0.01, 1, 1, 1, 0.4448235294117647, 0.01, 0.01, 0.01, 0.01, 0.01, 0.01, 0.01, 0.01, 0.01, 0.01, 0.01, 0.01, 0.01, 0.01, 0.01, 0.01, 0.01, 0.01, 0.01, 0.01, 0.01, 0.01, 0.01, 1, 0.926235294117647, 1, 0.9650588235294117, 0.01, 0.01, 0.01, 0.01, 0.01, 0.01, 0.01, 0.01, 0.01, 0.01, 0.01, 0.01, 0.01, 0.01, 0.01, 0.01, 0.01, 0.01, 0.01, 0.01, 0.01, 0.01, 0.01, 0.9029411764705882, 1, 1, 1, 0.01, 0.01, 0.01, 0.01, 0.01, 0.01, 0.01, 0.01, 0.01, 0.01, 0.01, 0.01, 0.01, 0.01, 0.01, 0.01, 0.01, 0.01, 0.01, 0.01, 0.01, 0.01, 0.01, 0.044941176470588234, 1, 1, 1, 0.01, 0.01, 0.01, 0.01, 0.01, 0.01, 0.01, 0.01, 0.01, 0.01, 0.01, 0.01, 0.01, 0.01, 0.01, 0.01, 0.01, 0.01, 0.01, 0.01, 0.01, 0.01, 0.01, 0.01, 0.9107058823529413, 1, 0.9961176470588236, 0.01, 0.01, 0.01, 0.01, 0.01, 0.01, 0.01, 0.01, 0.01, 0.01, 0.01, 0.01, 0.01, 0.01, 0.01, 0.01, 0.01, 0.01, 0.01, 0.01, 0.01, 0.01, 0.01, 0.01, 0.01, 0.8369411764705883, 1, 0.23905882352941177, 0.01, 0.01, 0.01, 0.01, 0.01, 0.01, 0.01, 0.01, 0.01, 0.01, 0.01, 0.01, 0.01, 0.01, 0.01, 0.01, 0.01, 0.01, 0.01, 0.01, 0.01, 0.01, 0.01, 0.01, 0.8912941176470588, 0.9961176470588236, 1, 0.01, 0.01, 0.01, 0.01, 0.01, 0.01, 0.01, 0.01, 0.01, 0.01, 0.01, 0.01, 0.01, 0.01, 0.01, 0.01, 0.01, 0.01, 0.01, 0.01, 0.01, 0.01, 0.01, 0.01, 0.01, 1, 1, 1, 1, 1, 1, 0.8563529411764705, 0.15364705882352944, 0.01, 0.01, 0.01, 0.01, 0.01, 0.01, 0.01, 0.01, 0.01, 0.01, 0.01, 0.01, 0.01, 0.01, 0.01, 0.01, 0.01, 0.01, 0.01, 0.01, 0.01, 1, 1, 1, 1, 0.9223529411764706, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 0.01, 0.01, 0.01, 0.01, 0.01, 0.01, 0.01, 0.01, 0.01, 0.01, 0.01, 0.01, 0.01, 0.01, 0.01, 0.7437647058823529, 1, 1, 1, 0.7825882352941176, 1, 1, 1, 1, 1, 1, 1, 1, 0.01, 0.01, 0.01, 0.01, 0.01, 0.01, 0.01, 0.01, 0.01, 0.01, 0.01, 0.01, 0.01, 0.01, 0.01, 0.01, 0.01, 0.01, 0.01, 0.01, 0.01, 0.13423529411764706, 0.4370588235294118, 0.5884705882352941, 0.5884705882352941, 0.5884705882352941, 0.5884705882352941, 0.5884705882352941, 0.01, 0.01, 0.01, 0.01, 0.01, 0.01, 0.01, 0.01, 0.01, 0.01, 0.01, 0.01, 0.01, 0.01, 0.01, 0.01, 0.01, 0.01, 0.01, 0.01, 0.01, 0.01, 0.01, 0.01, 0.01, 0.01, 0.01, 0.01, 0.01, 0.01, 0.01, 0.01, 0.01, 0.01, 0.01, 0.01, 0.01, 0.01, 0.01, 0.01, 0.01, 0.01, 0.01, 0.01, 0.01, 0.01, 0.01, 0.01, 0.01, 0.01, 0.01, 0.01, 0.01, 0.01, 0.01, 0.01, 0.01, 0.01, 0.01, 0.01, 0.01, 0.01, 0.01, 0.01, 0.01, 0.01, 0.01, 0.01, 0.01, 0.01, 0.01, 0.01, 0.01, 0.01, 0.01, 0.01, 0.01, 0.01, 0.01, 0.01, 0.01, 0.01, 0.01, 0.01, 0.01, 0.01, 0.01, 0.01, 0.01, 0.01, 0.01, 0.01, 0.01, 0.01, 0.01, 0.01, 0.01, 0.01, 0.01, 0.01, 0.01, 0.01, 0.01, 0.01, 0.01, 0.01, 0.01, 0.01, 0.01, 0.01, 0.01, 0.01, 0.01, 0.01, 0.01, 0.01, 0.01, 0.01, 0.01, 0.01, 0.01, 0.01, 0.01, 0.01, 0.01, 0.01, 0.01, 0.01, 0.01, 0.01, 0.01, 0.01, 0.01, 0.01, 0.01, 0.01, 0.01, 0.01, 0.01, 0.01, 0.01, 0.01, 0.01, 0.01]
    #arr = np.array(input)
    #result = nn.singleResult(arr)
    #print(result)

    



def truncated_normal(mean=0.00, sd=0.2, a=-1.0, b=1.0):
    #set parameters for a truncated normal distribution, from which initialized weights in the range [a,b] will be randomly generated
    return truncnorm((a - mean) / sd, (b - mean) / sd, loc=mean, scale=sd)

@np.vectorize
def sigmoid(signal):
    return 1 / (1 + np.e ** -signal)
    
@np.vectorize
def sigmoid_derivative(x):
    num = np.e**-x
    den = (1 + np.e ** -x) ** 2
    return num/den

class NeuralNetwork:
    def __init__(self, network_structure, # ie. [input_nodes, hidden_1_nodes, ... , hidden_n_nodes, output_nodes]
                 learning_rate, weights_matrices):  
            
        self.structure = network_structure
        self.learning_rate = learning_rate 
        if weights_matrices == None:
            self.weights_matrices = []
            self.populate_weight_matrices()
        else:
            self.weights_matrices = weights_matrices
        self.activation_vectors = []
        self.z_vectors = []
        self.delta_vectors = []

    #initialize weight matrices with random numbers and add each matrix to array self.weights_matrices
    def populate_weight_matrices(self):        
        #an array of weights matrices, which will be of length layers, L, to which we will add the weights matrix for each layer  
          
        layer_index = 1
        L = len(self.structure)-1
        while layer_index <= L:
            nodes_in = self.structure[layer_index-1]
            nodes_out = self.structure[layer_index]
            
            #calculate total number of elements in matrix
            #add 1 to nodes_in to account for the threshold value, which will be included as a weight from a node with a token value of 1
            n = (nodes_in+1) * nodes_out 
            
            #create weight matrix W and add to network's array of weight matrices
            tn = truncated_normal()
            #W = [np.random.randn(y, x)/np.sqrt(x) 
                        #for x, y in zip(self.sizes[:-1], self.sizes[1:])]
            W = tn.rvs(n).reshape((nodes_out, nodes_in+1)) 
            self.weights_matrices.append(W)  
            
            layer_index += 1
            
    def feedforward(self, x):
        x_vector = np.array(x, ndmin=2).T 
        x_vector = np.concatenate((x_vector, [[1]])) #we add a token 1 to the end of the vector to allow us to use a weight as a bias
        self.activation_vectors = [x_vector] 
        self.z_vectors = []
        
        layer_index = 0
        L = len(self.structure)-1
        while layer_index < L:
            in_vector = self.activation_vectors[layer_index]
            
            W = self.weights_matrices[layer_index]
            out_vector = np.dot(W, in_vector)
            
            self.z_vectors.append(out_vector)
            
            #use softmax, which can be interpreted as a probability, on the last layer
            if layer_index == L-1:
                out_vector = scipy.special.softmax(out_vector)
            #use sigmoid on all other layers
            else:
                out_vector = sigmoid(out_vector)
                
            if layer_index != L-1:
                out_vector = np.concatenate((out_vector, [[1]])) 
            self.activation_vectors.append(out_vector)
            
            layer_index+=1
            
        return self.activation_vectors[-1]
            
    def backprop(self, x, y):
        #BACKPROPAGATION
        y = np.array(y, ndmin=2).T 
        
        L = len(self.structure)-1
        
        z_vector_L = self.z_vectors[L-1]
        activation_vector_L = self.activation_vectors[L]
        
        #δ_L = ∇C_wrt_a ⊙ σ′(z_L)
              
        #cross-entropy cost 
        #cost function chosen such that σ′(z_L) term cancels
        delta_vector_L = (activation_vector_L - y)
        
        #quadratic cost 
        #delta_vector_L = (activation_vector_L - y) * sigmoid_derivative(z_vector_L)
        
        delta_vectors = [[] for _ in range(L)] #these will get populated in reverse order
        delta_vectors[L-1] = delta_vector_L
        
        layer_index = L-1
        while layer_index >= 1:
            W_transpose = self.weights_matrices[layer_index].transpose()
            last_delta_vector = delta_vectors[layer_index]
            
            if layer_index != L-1:
                last_delta_vector = last_delta_vector[:-1]
              
            #δ_l = ((w_l+1).T dot δ_l+1) ⊙ σ′(z_l)    
            tmp = np.dot(W_transpose, last_delta_vector)
            curr_z_vector = self.z_vectors[layer_index-1]
            curr_z_vector = sigmoid_derivative(curr_z_vector)
            curr_z_vector = np.concatenate((curr_z_vector, [[1]]))
            curr_delta_vector = tmp * curr_z_vector
            
            delta_vectors[layer_index-1] = curr_delta_vector
            layer_index-=1
            
        #GRADIENT DESCENT
        partial_cost_wrt_w = []
        
        layer_index = 1
        while layer_index <= L:
            delta_vector = delta_vectors[layer_index-1]
            activation_vector = self.activation_vectors[layer_index-1]
            
            if layer_index != L:
                delta_vector = delta_vector[:-1]
            
            #∂C/∂w = a_in dot δ_out
            tmp = np.dot(activation_vector, delta_vector.transpose()) #partial derivatives of the cost function wrt weights at this layer
            
            partial_cost_wrt_w.append(tmp.transpose()) 
            
            layer_index+=1
         
        self.weights_matrices = [w-(self.learning_rate*(pc)) for w, pc in zip(self.weights_matrices, partial_cost_wrt_w)]

    @timer
    def train(self, train_imgs, train_labels_one_hot): 
        for x, y in zip(train_imgs, train_labels_one_hot):
            self.feedforward(x)
            self.backprop(x, y)
    
    def evaluate(self, test_imgs, test_labels_one_hot):
        corrects, wrongs = 0, 0
        for x, y in zip(test_imgs, test_labels_one_hot):
            res = self.feedforward(x)
            res_max = res.argmax() #capture network's best guess, ie. highest valued neuron in last layer
            if res_max == y.argmax():
                corrects += 1
            else:
                wrongs += 1
                
        return corrects, wrongs
    
    @timer
    def processData(self): 
        image_size = 28 # width and length
        no_of_different_labels = 10 #  i.e. 0, 1, 2, 3, ..., 9
        image_pixels = image_size**2
        train_data = np.loadtxt(data_path + "mnist_train.csv", delimiter=",")
        test_data = np.loadtxt(data_path + "mnist_test.csv", delimiter=",") 
            
        fac = 0.99 / 255
        train_imgs = np.asfarray(train_data[:, 1:]) * fac + 0.01
        test_imgs = np.asfarray(test_data[:, 1:]) * fac + 0.01
            
        train_labels = np.asfarray(train_data[:, :1])
        test_labels = np.asfarray(test_data[:, :1])
            
        lr = np.arange(no_of_different_labels)
        
            # transform labels into one hot representation
        train_labels_one_hot = (lr==train_labels).astype(np.float)
        test_labels_one_hot = (lr==test_labels).astype(np.float)
            
            # we don't want zeroes and ones in the labels either:
        train_labels_one_hot[train_labels_one_hot==0] = 0.01
        train_labels_one_hot[train_labels_one_hot==1] = 0.99
        test_labels_one_hot[test_labels_one_hot==0] = 0.01
        test_labels_one_hot[test_labels_one_hot==1] = 0.99
            
        with open(data_path + "pickled_mnist.pkl", "bw") as fh:
            data = (train_imgs, 
                        test_imgs, 
                        train_labels,
                        test_labels,
                        train_labels_one_hot,
                        test_labels_one_hot)
            pickle.dump(data, fh)
              
    def loadPickle(self):
        with open(data_path + "pickled_mnist.pkl", "br") as fh:
            data = pickle.load(fh)
            
        train_imgs = data[0]
        test_imgs = data[1]
        train_labels = data[2]
        test_labels = data[3]
        train_labels_one_hot = data[4]
        test_labels_one_hot = data[5]
            
        return data
        
    @timer 
    #network_structure should be [input_nodes, hidden_1_nodes, ... , hidden_n_nodes, output_nodes]
    #learning_rate is the rate of gradient descent
    #num_runs determines the number of passes through the same training data this method will use
    def trainNetwork(self):
        data = self.loadPickle()
        train_imgs = data[0]
        train_labels_one_hot = data[4]
        test_imgs = data[1]
        test_labels_one_hot = data[5]
            
        for i in range(num_runs):
            self.train(train_imgs, train_labels_one_hot)
            print('Finished training run', i+1)
                #corrects, wrongs = nn.evaluate(test_imgs, test_labels_one_hot)
                #print('Training run %2d -- Percent correct: %.3f' % (i+1, corrects/(corrects+wrongs)))
                
        corrects, wrongs = self.evaluate(test_imgs, test_labels_one_hot)
        #print("Testing -- Number correct: %d" % corrects)
            #print("Testing -- Number wrong: %d" % wrongs)
        print('Testing -- Percent correct: %.3f' % (corrects/(corrects+wrongs)))
        
        weights_matrices = self.weights_matrices
        self.storeWeights(weights_matrices)
            
    def storeWeights(self, weights_matrices):
        with open(data_path + "pickled_weights.pkl", "bw") as fh:
            data = weights_matrices
            pickle.dump(data, fh)
            
    @timer 
    def singleResult(self, arr):
        with open(data_path + "pickled_weights.pkl", "br") as fh:
            data = pickle.load(fh)
        weights_matrices = data
            
        nn = NeuralNetwork(network_structure, learning_rate, weights_matrices)
            
            #get the final activation signals and process them into an array
        final_activations = nn.feedforward(arr)
        result = []
        for i in range(len(final_activations)):
            result.append(final_activations[i][0])
                
        return result


   
if __name__ == '__main__':
    main()

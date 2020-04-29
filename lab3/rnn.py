import numpy as np

class RNN():
    def __init__(self, hidden_size, sequence_length, vocab_size, learning_rate):
        self.hidden_size = hidden_size
        self.sequence_length = sequence_length
        self.vocab_size = vocab_size
        self.learning_rate = learning_rate

        self.U = None # ... input projection
        self.W = None # ... hidden-to-hidden projection
        self.b = None # ... input bias

        self.V = None # ... output projection
        self.c = None # ... output bias

        # memory of past gradients - rolling sum of squares for Adagrad
        self.memory_U, self.memory_W, self.memory_V = np.zeros_like(self.U), np.zeros_like(self.W), np.zeros_like(self.V)
        self.memory_b, self.memory_c = np.zeros_like(self.b), np.zeros_like(self.c)


    def rnn_step_forward(self, x, h_prev, U, W, b):
        # A single time step forward of a recurrent neural network with a 
        # hyperbolic tangent nonlinearity.

        # x - input data (minibatch size x input dimension)
        # h_prev - previous hidden state (minibatch size x hidden size)
        # U - input projection matrix (input dimension x hidden size)
        # W - hidden to hidden projection matrix (hidden size x hidden size)
        # b - bias of shape (hidden size x 1)

        h_current, cache = None, None
        
        # return the new hidden state and a tuple of values needed for the backward step

        return h_current, cache


    def rnn_forward(self, x, h0, U, W, b):
        # Full unroll forward of the recurrent neural network with a 
        # hyperbolic tangent nonlinearity

        # x - input data for the whole time-series (minibatch size x sequence_length x input dimension)
        # h0 - initial hidden state (minibatch size x hidden size)
        # U - input projection matrix (input dimension x hidden size)
        # W - hidden to hidden projection matrix (hidden size x hidden size)
        # b - bias of shape (hidden size x 1)
        
        h, cache = None, None

        # return the hidden states for the whole time series (T+1) and a tuple of values needed for the backward step

        return h, cache


    def rnn_step_backward(self, grad_next, cache):
        # A single time step backward of a recurrent neural network with a 
        # hyperbolic tangent nonlinearity.

        # grad_next - upstream gradient of the loss with respect to the next hidden state and current output
        # cache - cached information from the forward pass

        dh_prev, dU, dW, db = None, None, None, None
        
        # compute and return gradients with respect to each parameter
        # HINT: you can use the chain rule to compute the derivative of the
        # hyperbolic tangent function and use it to compute the gradient
        # with respect to the remaining parameters

        return dh_prev, dU, dW, db


    def rnn_backward(self, dh, cache):
        # Full unroll forward of the recurrent neural network with a 
        # hyperbolic tangent nonlinearity
        
        dU, dW, db = None, None, None

        # compute and return gradients with respect to each parameter
        # for the whole time series.
        # Why are we not computing the gradient with respect to inputs (x)?

        return dU, dW, db

    def output(h, V, c):
        # Calculate the output probabilities of the network
        pass

    def output_loss_and_grads(self, h, V, c, y):
        # Calculate the loss of the network for each of the outputs
        
        # h - hidden states of the network for each timestep. 
        #     the dimensionality of h is (batch size x sequence length x hidden size (the initial state is irrelevant for the output)
        # V - the output projection matrix of dimension hidden size x vocabulary size
        # c - the output bias of dimension vocabulary size x 1
        # y - the true class distribution - a tensor of dimension 
        #     batch_size x sequence_length x vocabulary size - you need to do this conversion prior to
        #     passing the argument. A fast way to create a one-hot vector from
        #     an id could be something like the following code:

        #   y[batch_id][timestep] = np.zeros((vocabulary_size, 1))
        #   y[batch_id][timestep][batch_y[timestep]] = 1

        #     where y might be a list or a dictionary.

        loss, dh, dV, dc = None, None, None, None
        # calculate the output (o) - unnormalized log probabilities of classes
        # calculate yhat - softmax of the output
        # calculate the cross-entropy loss
        # calculate the derivative of the cross-entropy softmax loss with respect to the output (o)
        # calculate the gradients with respect to the output parameters V and c
        # calculate the gradients with respect to the hidden layer h

        return loss, dh, dV, dc

    
    def update(self, dU, dW, db, dV, dc, U, W, b, V, c, memory_U, memory_W, memory_b, memory_V, memory_c):
        # update memory matrices
        # perform the Adagrad update of parameters
        pass




def run_language_model(dataset, max_epochs, hidden_size=100, sequence_length=30, learning_rate=1e-1, sample_every=100):
    
    vocab_size = len(dataset.sorted_chars)
    RNN = None # initialize the recurrent network

    current_epoch = 0 
    batch = 0

    h0 = np.zeros((hidden_size, 1))

    average_loss = 0

    while current_epoch < max_epochs: 
        e, x, y = dataset.next_minibatch()
        
        if e: 
            current_epoch += 1
            h0 = np.zeros((hidden_size, 1))
            # why do we reset the hidden state here?

        # One-hot transform the x and y batches
        x_oh, y_oh = None, None

        # Run the recurrent network on the current batch
        # Since we are using windows of a short length of characters,
        # the step function should return the hidden state at the end
        # of the unroll. You should then use that hidden state as the
        # input for the next minibatch. In this way, we artificially
        # preserve context between batches.
        loss, h0 = RNN.step(h0, x_oh, y_oh)

        if batch % sample_every == 0: 
            # run sampling (2.2)
            pass
        batch += 1


def sample(seed, n_sample):
    h0, seed_onehot, sample = None, None, None 
    # inicijalizirati h0 na vektor nula
    # seed string pretvoriti u one-hot reprezentaciju ulaza
    
    return sample
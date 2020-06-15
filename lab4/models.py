class RBM():
    
    def __init__(self, visible_size, hidden_size, cd_k=1):
        self.v_size = visible_size
        self.h_size = hidden_size
        self.cd_k = cd_k
        
        normal_dist = tdist.Normal(0, 0.1)
        
        self.W = torch.Tensor(normal_dist.sample(sample_shape=(self.v_size, self.h_size))).cuda()
        self.v_bias = torch.Tensor(torch.zeros(self.v_size)).cuda()
        self.h_bias = torch.Tensor(torch.zeros(self.h_size)).cuda()


    def __call__(self, batch):
        return self.forward(batch)

    
    def forward(self, batch):
        return self._cd_pass(batch)
    
    
    def _cd_pass(self, batch):
        batch = batch.view(-1, 784)
        v0 = torch.bernoulli(batch)
        h0_prob = sigmoid(torch.matmul(v0, self.W) + self.h_bias)
        h0 = torch.bernoulli(h0_prob)

        h1 = h0

        for step in range(0, self.cd_k):
            v1_prob = sigmoid(torch.matmul(h1, self.W.T) + self.v_bias)
            v1 = torch.bernoulli(v1_prob)
            h1_prob = sigmoid(torch.matmul(v1, self.W) + self.h_bias)
            h1 = torch.bernoulli(h1_prob)
            
        return h0_prob, h0, h1_prob, h1, v1_prob, v1
    
    def reconstruct(self, h, gibbs_steps=None):
        h1 = h
        
        steps_to_do = self.cd_k
        if gibbs_steps is not None:
            steps_to_do = gibbs_steps

        for step in range(0, steps_to_do):
            v1_prob = sigmoid(torch.matmul(h1, self.W.T) + self.v_bias)
            v1 = torch.bernoulli(v1_prob)
            h1_prob = sigmoid(torch.matmul(v1, self.W) + self.h_bias)
            h1 = torch.bernoulli(h1_prob)

        return h1_prob, h1, v1_prob, v1

    
    def update_weights_for_batch(self, batch, learning_rate=0.01):
        h0_prob, h0, h1_prob, h1, v1_prob, v1 = self._cd_pass(batch)
        v0 = torch.bernoulli(batch)

        w_positive_grad = torch.matmul(v0.T, h0)
        w_negative_grad = torch.matmul(v1.T, h1)

        dw = (w_positive_grad - w_negative_grad) / batch.shape[0]

        self.W = self.W + learning_rate * dw
        self.v_bias = self.v_bias + learning_rate*torch.mean(v0-v1, dim=0)
        self.h_bias = self.h_bias + learning_rate*torch.mean(h0-h1, dim=0)


class DBN():

    def __init__(self, first_rbm: RBM, second_hidden_size, cd_k=1):
        self.v_size = first_rbm.v_size
        self.h1_size = first_rbm.h_size
        self.h2_size = second_hidden_size
        self.cd_k = cd_k
        
        normal_dist = tdist.Normal(0, 0.1)
        
        self.W1 = first_rbm.W.cuda()
        self.v_bias = first_rbm.v_bias.clone().cuda()
        self.h1_bias = first_rbm.h_bias.clone().cuda()
        
        self.W2 = torch.Tensor(normal_dist.sample(sample_shape=(self.h1_size, self.h2_size))).cuda()
        self.h2_bias = torch.Tensor(torch.zeros(self.h2_size)).cuda()
    
    
    def forward(self, batch, steps=None):
        batch = batch.view(-1, 784)
        # v = torch.bernoulli(batch)
        h1up_prob = sigmoid(torch.matmul(batch, self.W1) + self.h1_bias)
        h1up = torch.bernoulli(h1up_prob)
        
        h2up_prob = sigmoid(torch.matmul(h1up, self.W2) + self.h2_bias)
        h2up = torch.bernoulli(h2up_prob)
        
        h1down_prob, h1down, h2down_prob, h2down = self.gibbs_sampling(h2up, steps)
        
        return h1up_prob, h1up, h2up_prob, h2up, h1down_prob, h1down, h2down_prob, h2down

    
    def gibbs_sampling(self, h2, steps=None):
        h2down = h2
        
        steps_to_do = self.cd_k
        
        if steps is not None:
            steps_to_do = steps

        for step in range(0, steps_to_do):
            h1down_prob = sigmoid(torch.matmul(h2down, self.W2.T) + self.h1_bias)
            h1down = torch.bernoulli(h1down_prob)
            
            h2down_prob = sigmoid(torch.matmul(h1down, self.W2) + self.h2_bias)
            h2down = torch.bernoulli(h2down_prob)
            
        return h1down_prob, h1down, h2down_prob, h2down 
    
    def reconstruct(self, h2, steps=None):
        _, _, h2down_prob, h2down = self.gibbs_sampling(h2, steps)
        
        h1down_prob = sigmoid(torch.matmul(h2down, self.W2.T) + self.h1_bias)
        h1down = torch.bernoulli(h1down_prob)
        
        v_prob = sigmoid(torch.matmul(h1down, self.W1.T) + self.v_bias)
        v_out = torch.bernoulli(v_prob)
        
        return v_prob, v_out, h2down_prob, h2down
    
    def update_weights_for_batch(self, batch, learning_rate=0.01):
        h1up_prob, h1up, h2up_prob, h2up, h1down_prob, h1down, h2down_prob, h2down = self.forward(batch)

        w2_positive_grad = h2up * (h1up - h1down_prob)
        w2_negative_grad = h1down * (h2down - h2down_prob)

        dw2 = (w2_positive_grad - w2_negative_grad) / h1up.shape[0]

        self.W2 = self.W2 + learning_rate * dw2
        self.h1_bias = self.h1_bias + learning_rate*torch.mean(h1up - h1down, dim=0)
        self.h2_bias = self.h2_bias + learning_rate*torch.mean(h2up - h2down, dim=0)
                 
    
    def __call__(self, batch):
        return self.forward(batch)
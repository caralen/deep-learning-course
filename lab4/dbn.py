class DBN():

    def __init__(self, first_rbm: RBM, second_hidden_size, cd_k=1):
        self.v_size = first_rbm.v_size
        self.h1_size = first_rbm.h_size
        self.h2_size = second_hidden_size
        self.cd_k = cd_k
        
        normal_dist = tdist.Normal(0, 0.1)
        
        self.W1 = first_rbm.W
        self.v_bias = first_rbm.v_bias.clone()
        self.h1_bias = first_rbm.h_bias.clone()
        
        self.W2 = torch.Tensor(normal_dist.sample(sample_shape=(self.h1_size, self.h2_size)))
        self.h2_bias = torch.Tensor(torch.zeros(self.h2_size))
    
    
    def forward(self, batch, steps=None):
        batch = batch.view(-1, 784)
        
        h1up_prob = 
        h1up = 
        
        h2up_prob = 
        h2up = 
        
        h1down_prob, h1down, h2down_prob, h2down = self.gibbs_sampling(???, steps)
        
        return h1up_prob, h1up, h2up_prob, h2up, h1down_prob, h1down, h2down_prob, h2down

    
    def gibbs_sampling(self, h2, steps=None):
        h2down = h2
        
        steps_to_do = self.cd_k
        
        if steps is not None:
            steps_to_do = steps

        for step in range(0, steps_to_do):
            h1down_prob = 
            h1down = 
            
            h2down_prob = 
            h2down = 
            
        return h1down_prob, h1down, h2down_prob, h2down 
    
    def reconstruct(self, h2, steps=None):
        _, _, h2down_prob, h2down = self.gibbs_sampling(???, steps)
        
        h1down_prob = 
        h1down = 
        
        v_prob = 
        v_out = 
        
        return v_prob, v_out, h2down_prob, h2down
    
    def update_weights_for_batch(self, batch, learning_rate=0.01):
        h1up_prob, h1up, h2up_prob, h2up, h1down_prob, h1down, h2down_prob, h2down = self.forward(batch)

        w2_positive_grad = 
        w2_negative_grad = 

        dw2 = (w2_positive_grad - w2_negative_grad) / h1up.shape[0]

        self.W2 = self.W2 + 
        self.h1_bias = self.h1_bias + 
        self.h2_bias = self.h2_bias + 
        
                
    
    def __call__(self, batch):
        return self.forward(batch)


dbnmodel = DBN(model, second_hidden_size=100, cd_k=2)
for curr_epoch in tqdm.tqdm(range(0, EPOCHS)):
    for sample, label in train_loader:
        sample = sample.view(-1, 784)
        dbnmodel.update_weights_for_batch(sample, learning_rate=0.1)

plt.figure(figsize=(12, 12), facecolor='w')
visualize_RBM_weights(dbnmodel.W2.data.cpu(), 10, 10, slice_shape=(10, 10))

sample, _ = next(iter(test_loader))
sample = sample.view(-1, 784)

for idx in range(0, 20):
    h1up_prob, h1up, h2up_prob, h2up, h1down_prob, h1down, h2down_prob, h2down = dbnmodel(sample[idx, ...])
    v_prob, v, _, _ = dbnmodel.reconstruct(h2down)

    plt.figure(figsize=(4*3, 4))
    plt.subplot(1, 3, 1)
    plt.imshow(sample[idx,...].view(28, 28))
    if idx == 0:
        plt.title("Test input")
    
    plt.subplot(1, 3, 2)
    plt.imshow(v_prob[0, ...].view(28, 28))
    if idx == 0:
        plt.title("Reconstruction")
    
    plt.subplot(1, 3, 3)
    plt.imshow(h2down.view(10, 10))
    if idx == 0:
        plt.title("Hidden state")


r_input = np.random.rand(100, HIDDEN_SIZE)
r_input[r_input > 0.9] = 1 # postotak aktivnih - slobodno varirajte
r_input[r_input < 1] = 0
r_input = r_input * 20 # pojačanje za slučaj ako je mali postotak aktivnih

s = 10
i = 0
r_input[i,:] = 0
r_input[i,i]= s
i += 1
r_input[i,:] = 0
r_input[i,i]= s
i += 1
r_input[i,:] = 0
r_input[i,i]= s
i += 1
r_input[i,:] = 0
r_input[i,i]= s
i += 1
r_input[i,:] = 0
r_input[i,i]= s
i += 1
r_input[i,:] = 0
r_input[i,i]= s
i += 1
r_input[i,:] = 0
r_input[i,i]= s

v_out_prob, v_out, h2down_prob, h2down = dbnmodel.reconstruct(torch.from_numpy(r_input).float(), 100)

plt.figure(figsize=(16, 16))
for idx in range(0, 19):
    plt.figure(figsize=(14, 4))
    
    plt.subplot(1, 3, 1)
    plt.imshow(r_input[idx, ...].reshape(10, 10))
    if idx == 0:
        plt.title("Set state")
    plt.axis('off')
    
    plt.subplot(1, 3, 2)
    plt.imshow(h2down[idx, ...].view(10, 10))
    if idx == 0:
        plt.title("Final state")
    plt.axis('off')
    
    plt.subplot(1, 3, 3)
    plt.imshow(v_out_prob[idx, ...].view(28, 28))
    if idx == 0:
        plt.title("Reconstruction")
    plt.axis('off')
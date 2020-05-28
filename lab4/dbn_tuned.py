class DBNWithFineTuning():

    def __init__(self, base_dbn: DBN, cd_k=1):
        self.v_size = base_dbn.v_size
        self.h1_size = base_dbn.h1_size
        self.h2_size = base_dbn.h2_size
        self.cd_k = cd_k
        
        normal_dist = tdist.Normal(0, 0.1)
        
        self.R1 = base_dbn.W1.clone()
        self.W1_down = base_dbn.W1.T.clone()
        self.v1_bias = base_dbn.v_bias.clone()
        self.h1_up_bias = base_dbn.h1_bias.clone()
        self.h1_down_bias = base_dbn.h1_bias.clone()
        
        self.W2 = base_dbn.W2.clone()
        self.h2_bias = base_dbn.h2_bias.clone()
    
    
    def forward(self, batch, steps=None):
        batch = batch.view(-1, 784)
        
        h1_up_prob = 
        h1_up = 
        
        v1_up_down_prob = 
        v1_up_down = 
        
        h2_up_prob = 
        h2_up = 
        
        h1_down_prob, h1_down, h2_down_prob, h2_down = self.gibbs_sampling(???, steps=steps)
        
        v1_down_prob = 
        v1_down = 
        
        h1_down_up_prob = 
        h1_down_up = 
        
        return h1_up_prob, h1_up, v1_up_down_prob, v1_up_down, h2_up_prob, h2_up, h1_down_prob, h1_down, h2_down_prob, h2_down, v1_down_prob, v1_down, h1_down_up_prob, h1_down_up
    
    def gibbs_sampling(self, h2, steps=None):
        h2_down = h2
        
        steps_to_do = self.cd_k
        
        if steps is not None:
            steps_to_do = steps
        
        
        for step in range(0, self.cd_k):
            h1_down_prob =
            h1_down = 

            h2_down_prob = 
            h2_down = 
            
        return h1_down_prob, h1_down, h2_down_prob, h2_down


    
    def reconstruct(self, h2, steps=None):
        h1_down_prob, h1_down, h2_down_prob, h2down = self.gibbs_sampling(???, steps)
        
        v_out_tmp_prob = 
        v_out_tmp =
        v_out_prob = 
        v_out = 
        
        return v_out_prob, v_out, h2_down_prob, h2down
    
    def update_weights_for_batch(self, batch, learning_rate=0.01):
        h1_up_prob, h1_up, v1_up_down_prob, v1_up_down, h2_up_prob, h2_up, h1_down_prob, h1_down, h2_down_prob, h2_down, v1_down_prob, v1_down, h1_down_up_prob, h1_down_up = self.forward(batch)
        
        self.W1_down = self.W1_down + 
        self.R1 = self.R1 + 
        
        self.v1_bias = self.v1_bias +
        
        self.h1_down_bias = self.h1_down_bias + 
        self.h1_up_bias = self.h1_up_bias + 
        
        
        w2_positive_grad = 
        w2_negative_grad = 
        dw2 = (w2_positive_grad - w2_negative_grad) / h1_up.shape[0]
        
        self.W2 = self.W2 + 
        self.h2_bias = self.h2_bias + 
        
    
    def __call__(self, batch):
        return self.forward(batch)


dbnmodel_ft = DBNWithFineTuning(dbnmodel, cd_k=2)
for curr_epoch in tqdm.tqdm(range(0, EPOCHS)):
    for sample, label in train_loader:
        sample = sample.view(-1, 784)
        dbnmodel_ft.update_weights_for_batch(sample, 0.01)

plt.figure(figsize=(12, 12), facecolor='w')
visualize_RBM_weights(dbnmodel_ft.R1.data, 10, 10)
plt.tight_layout()


plt.figure(figsize=(12, 12), facecolor='w')
visualize_RBM_weights(dbnmodel_ft.W1_down.T.data, 10, 10)
plt.tight_layout()

difference = torch.abs(dbnmodel_ft.R1.data - dbnmodel_ft.W1_down.T.data)
plt.figure(figsize=(12, 12), facecolor='w')
visualize_RBM_weights(difference, 10, 10)
plt.tight_layout()

sample, _ = next(iter(test_loader))
sample = sample.view(-1, 784)

for idx in range(0, 20): 
    # rbn reconstruct
    _, _, _, _, recon1, _ = model(sample[idx, ...])
    
    # dbn reconstruct
    _, _, _, _, _, _, _, h2down = dbnmodel.forward(sample[idx, ...])
    recon2, _, _, _ = dbnmodel.reconstruct(h2down)
    
    # dbn fine tune reconstruct
    _, _, _, _, _, _, _, _, _, h2_down, _, _, _, _ = dbnmodel_ft(sample[idx, ...])
    recon3, _, _, _ = dbnmodel_ft.reconstruct(h2_down, 2)
    
    plt.figure(figsize=(5*3, 3))
    plt.subplot(1, 5, 1)
    plt.imshow(sample[idx, ...].view(28, 28))
    if idx == 0:
        plt.title("Original image")
    
    plt.subplot(1, 5, 2)
    plt.imshow(recon1.view(28, 28))
    if idx == 0:
        plt.title("Reconstruction 1")
    
    plt.subplot(1, 5, 3)
    plt.imshow(recon2.view(28, 28))
    if idx == 0:
        plt.title("Reconstruction 2")
    
    plt.subplot(1, 5, 4)
    plt.imshow(recon3.view(28, 28))
    if idx == 0:
        plt.title("Reconstruction 3")
    
    plt.subplot(1, 5, 5)
    plt.imshow(h2_down.view(10, 10))
    if idx == 0:
        plt.title("Top state 3")

r_input = np.random.rand(100, 100)
r_input[r_input > 0.9] = 1
r_input[r_input < 1] = 0

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

    
v_out_prob, v_out, h2_down_prob, h2down = dbnmodel_ft.reconstruct(torch.from_numpy(r_input).float(), 100)

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
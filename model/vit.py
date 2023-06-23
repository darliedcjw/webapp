import torch
import torch.nn as nn
import numpy as np

'''
Batchnorm2d calculates mean and sd with respect to each channel for the batch
LayerNorm calculates mean and sd with respect to last D dimensions of normalized shape
'''

'''
hstack: concatenate column wise
vstack: concatenate row wise
stack: concatenate batch
'''

device = torch.device('cuda')

class MSA(nn.Module):
    def __init__(self, hidden_d, n_heads=2):
        super(MSA, self).__init__()
        self.hidden_d = hidden_d
        self.n_heads = n_heads

        # Number of heads for different standards of computation
        assert hidden_d % n_heads == 0 

        self.d_head = int(hidden_d / n_heads)
        # nn.Linear is a module type and is therefore stored as a module list
        self.q_mappings = nn.ModuleList([nn.Linear(self.d_head, self.d_head) for _ in range(self.n_heads)])
        self.k_mappings = nn.ModuleList([nn.Linear(self.d_head, self.d_head) for _ in range(self.n_heads)])
        self.v_mappings = nn.ModuleList([nn.Linear(self.d_head, self.d_head) for _ in range(self.n_heads)])
        self.softmax = nn.Softmax(dim=-1)

    def forward(self, images):
        '''
        Sequences: N x Seq Len x Token Dim
        Multi-Head: N x Seq Len x Token Dim / n_heads

        Patchs: B x Patches x Flatten Pixel
        **Flatten Pixel is treated as token Token Dim
        '''
        result = []

        for image in images:
            image_result = []
            for head in range(self.n_heads):
                q_mapping = self.q_mappings[head]
                k_mapping = self.k_mappings[head]
                v_mapping = self.v_mappings[head]

                # Working on splitting the head on each patch
                patches = image[:, head * self.d_head: (head + 1) * self.d_head]
                q, k, v = q_mapping(patches), k_mapping(patches), v_mapping(patches)

                # Relevance: Dimension is 50 x 50 
                attention = self.softmax(q @ k.T / (self.d_head ** 0.5))
                # Giving the sum of weighted value: Dimension is 50 x 4
                image_result.append(attention @ v)
            result.append(torch.hstack(image_result))
        return torch.stack(result)
    
class EncoderBlock(nn.Module):
    def __init__(self, hidden_d, n_heads, mlp_ratio=4):
        super(EncoderBlock, self).__init__()
        self.hidden_d = hidden_d
        self.n_head = n_heads
        self.ln1 = nn.LayerNorm(hidden_d)
        self.mhsa = MSA(hidden_d=self.hidden_d, n_heads=self.n_head)
        self.ln2 = nn.LayerNorm(hidden_d)
        self.mlp = nn.Sequential(
            nn.Linear(hidden_d, mlp_ratio * hidden_d),
            nn.GELU(),
            nn.Linear(mlp_ratio * hidden_d, hidden_d)
        )

    def forward(self, x):
        out = x + self.mhsa(self.ln1(x))
        out = out + self.mlp(self.ln2(out))
        return out

class ViT(nn.Module):
    def __init__(self, image_res=(1, 28, 28), n_patches=7, hidden_d=8, n_blocks=2, n_heads=2, out_d=10):
        super(ViT, self).__init__()
        self.channel = image_res[0]
        self.height = image_res[1]
        self.width = image_res[2]
        self.n_patches = n_patches
        self.n_blocks = n_blocks
        self.n_heads = n_heads
        self.hidden_d = hidden_d

        assert self.height % n_patches == 0, 'Height not cleanly divided!'
        assert self.width % n_patches == 0, 'Width not cleanly divided!'
 
        self.patch_size = self.height // self.n_patches, self.width // self.n_patches

        # 1. Linear Mapping (Flatten)
        self.linear_input = self.patch_size[0] * self.patch_size[1] * self.channel
        self.linear = nn.Linear(self.linear_input, self.hidden_d)

        # 2. Learnable Classification Token
        self.class_token = nn.Parameter(torch.rand(1, self.hidden_d))

        # 3. Positional Embedding
        self.pos_embed = nn.Parameter(torch.tensor(self.get_positional_embeddings(self.n_patches ** 2 + 1, self.hidden_d)))
        self.pos_embed.requires_grad = False

        # 4. Transformer Encoder Blocks
        self.encoder_blocks = nn.ModuleList([EncoderBlock(hidden_d=self.hidden_d, n_heads=self.n_heads) for _ in range(self.n_blocks)])

        # 5. Classification MLP
        self.mlp = nn.Sequential(
            nn.Linear(self.hidden_d, out_d),
            nn.Softmax(dim=-1)
        )

    def patchify(self, images, n_patches):
        num_images, _, _, _ = images.shape

        assert self.height == self.width, 'Implemented only for square images for now!'
        
        # Gives Batch x Number of Patches x Number of Pixels in a Patch
        patches = torch.zeros(num_images, n_patches ** 2, self.height * self.width * self.channel//n_patches ** 2).to(device)

        # Iterating through each image in a batch
        for idx, image in enumerate(images):
            for y_patch in range(n_patches):
                for x_patch in range(n_patches):
                    patch = image[:, y_patch * self.patch_size[0]:(y_patch + 1) * self.patch_size[0], x_patch * self.patch_size[1]:(x_patch + 1) * self.patch_size[1]]
                    patches[idx, y_patch * n_patches + x_patch] = patch.flatten()
        
        return patches
    
    # Postional Function
    '''
    For each patch, for each pixel, get position
    '''
    def get_positional_embeddings(self, sequence_length, d):
        result = torch.ones(sequence_length, d)
        for i in range(sequence_length):
            for j in range(d):
                result[i][j] = np.sin(i / (10000 ** (j / d))) if j % 2 == 0 else np.cos(i / (10000 ** ((j - 1) / d)))
        return result

    def forward(self, images):
        patches = self.patchify(images, self.n_patches)
        tokens = self.linear(patches)
        tokens = torch.stack([torch.vstack((self.class_token, tokens[i])) for i in range(len(tokens))])
        out = tokens + self.pos_embed.repeat(images.shape[0], 1, 1)

        for block in self.encoder_blocks:
            out = block(out)
        
        out = self.mlp(out)

        out = out[:, 0]
        
        return out

if __name__ == '__main__':
    model = ViT(image_res=(3, 28, 28), n_patches=7, hidden_d=8, n_blocks=2, n_heads=2, out_d=10)
    x = torch.randn(24, 3, 28, 28)
    
    output = model(x)
    print(output.shape)
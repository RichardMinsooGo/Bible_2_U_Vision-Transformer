import torch
import torch.nn as nn
import torch.nn.functional as F
import torchvision.transforms as T
import numpy as np

from PIL import Image
import matplotlib.pyplot as plt
import seaborn as sns


# unit test on patching an image

# read image and resize to 128
image = Image.open('car.png').resize((128, 128))

# convert to numpy array 
x = np.array(image)


# An Image Is Worth 16x16 Words
P = 16   # patch size
C = 3    # number of channels (RGB)

# split image into patches using numpy
patches = x.reshape(x.shape[0]//P, P, x.shape[1]//P, P, C).swapaxes(1, 2).reshape(-1, P, P, C)

# flatten patches
x_p = np.reshape(patches, (-1, P * P * C))

# get number of patches
N = x_p.shape[0]

print('Image shape: ', x.shape)  # width, height, channel
print('Number of patches: {} with resolution ({}, {})'.format(N, P, P))
print('Patches shape: ', patches.shape)
print('Flattened patches shape: ', x_p.shape)

# visualize data
# 
# display image and patches side-by-side

fig = plt.figure()

gridspec = fig.add_gridspec(1, 2)
ax1 = fig.add_subplot(gridspec[0])
ax1.set(title='Image')

# display image 
ax1.imshow(x)

subgridspec = gridspec[1].subgridspec(8, 8, hspace=-0.8)

# display patches
for i in range(8):    # N = 64, 8x8 grid
    for j in range(8):
        num = i * 8 + j
        ax = fig.add_subplot(subgridspec[i, j])
        ax.set(xticks=[], yticks=[])
        ax.imshow(patches[num])

# visualize data
#
# display flattened patches

# display first 10 flattened patches up to 25 values 
heat_map = x_p[:10, :25]

yticklabels = ['patch ' + str(i + 1) for i in range(10)]

plt.title('First 10 Flattened Patches')
ax = sns.heatmap(heat_map,  
                 cmap=sns.light_palette("#a275ac", as_cmap=True),
                 xticklabels=False, yticklabels=yticklabels,
                 linewidths=0.01, linecolor='white'
                )

# unit test on patch embeddings

# dimensionality of patch embeddings
D = 384

# batch size
B = 1

# convert flattened patches to tensor
x_p = torch.Tensor(x_p)

# add batch dimension
x_p = x_p[None, ...]    

# weight matrix E
E = nn.Parameter(torch.randn(1, P * P * C, D))

patch_embeddings = torch.matmul(x_p , E)

assert patch_embeddings.shape == (B, N, D)
print(patch_embeddings.shape)

# unit test on class token 

# init class token
class_token = nn.Parameter(torch.randn(1, 1, D))

patch_embeddings = torch.cat((class_token, patch_embeddings), 1)

print(patch_embeddings.shape)
assert patch_embeddings.shape == (B, N + 1, D)

# unit test on position embedddings 

# position embeddings
E_pos = nn.Parameter(torch.randn(1, N + 1, D))

z0 = patch_embeddings + E_pos

print(z0.shape)
assert z0.shape == (B, N + 1, D)



import math

d_head    = 64
# Vit-base model configurations
n_layer = 6; hid_dim = 384; n_heads = 8; pf_dim=1024 
dropout_prob = 0.1

""" scale dot product attention """
class ScaledDotProductAttention(nn.Module):
    """Calculate the attention weights.
    query, key, value must have matching leading dimensions.
    key, value must have matching penultimate dimension, i.e.: seq_len_k = seq_len_v.
    The mask has different shapes depending on its type(padding or look ahead)
    but it must be broadcastable for addition.
    
    query, key, value의 leading dimensions은 동일해야 합니다.
    key, value 에는 일치하는 끝에서 두 번째 차원이 있어야 합니다(예: seq_len_k = seq_len_v).
    MASK는 유형에 따라 모양이 다릅니다(패딩 혹은 미리보기(=look ahead)).
    그러나 추가하려면 브로드캐스트할 수 있어야 합니다.

    Args:
        query: query shape == (batch_size, n_heads, seq_len_q, depth)
        key: key shape     == (batch_size, n_heads, seq_len_k, depth)
        value: value shape == (batch_size, n_heads, seq_len_v, depth_v)
        mask: Float tensor with shape broadcastable
              to (batch_size, n_heads, seq_len_q, seq_len_k). Defaults to None.

    Returns:
        output, attention_weights
    """
    def __init__(self):
        super().__init__()
        self.dropout = nn.Dropout(dropout_prob)
        
    def forward(self, query, key, value ):

        # Q와 K의 곱. 어텐션 스코어 행렬.
        matmul_qk = torch.matmul(query, torch.transpose(key,2,3))

        # 스케일링
        # dk의 루트값으로 나눠준다.
        dk = key.shape[-1]
        scaled_attention_logits = matmul_qk / math.sqrt(dk)

        # 마스킹. 어텐션 스코어 행렬의 마스킹 할 위치에 매우 작은 음수값을 넣는다.
        # 매우 작은 값이므로 소프트맥스 함수를 지나면 행렬의 해당 위치의 값은 0이 된다.
        # if mask is not None:
        #     scaled_attention_logits += (mask * -1e9)

        # 소프트맥스 함수는 마지막 차원인 key의 문장 길이 방향으로 수행된다.
        # attention weight : (batch_size, n_heads, query의 문장 길이, key의 문장 길이)
        attention_weights = F.softmax(scaled_attention_logits, dim=-1)

        # output : (batch_size, n_heads, query의 문장 길이, hid_dim/n_heads)
        output = torch.matmul(attention_weights, value)

        return output, attention_weights

""" multi head attention """
class MultiHeadAttentionLayer(nn.Module):
    
    def __init__(self):
        super(MultiHeadAttentionLayer, self).__init__()
        
        self.q_linear = nn.Linear(hid_dim, n_heads * d_head)
        self.k_linear = nn.Linear(hid_dim, n_heads * d_head)
        self.v_linear = nn.Linear(hid_dim, n_heads * d_head)
        self.scaled_dot_attn = ScaledDotProductAttention()
        self.output_MHA = nn.Linear(n_heads * d_head, hid_dim)
        self.dropout = nn.Dropout(dropout_prob)
    
    def forward(self, Q, K, V):
        batch_size = Q.size(0)
        
        # q : (bs, n_heads, n_q_seq, d_head)
        # k : (bs, n_heads, n_k_seq, d_head)
        # v : (bs, n_heads, n_v_seq, d_head)
        query = self.q_linear(Q).view(batch_size, -1, n_heads, d_head).transpose(1,2)
        key   = self.k_linear(K).view(batch_size, -1, n_heads, d_head).transpose(1,2)
        value = self.v_linear(V).view(batch_size, -1, n_heads, d_head).transpose(1,2)

        # (bs, n_heads, n_q_seq, n_k_seq)
        # attn_mask = attn_mask.unsqueeze(1).repeat(1, n_heads, 1, 1)

        # (bs, n_heads, n_q_seq, d_head), (bs, n_heads, n_q_seq, n_k_seq)
        scaled_attention, attn_prob = self.scaled_dot_attn(query, key, value)
        
        # (bs, n_heads, n_q_seq, h_head * d_head)
        concat_attention = scaled_attention.transpose(1, 2).contiguous().view(batch_size, -1, n_heads * d_head)
        
        # (bs, n_heads, n_q_seq, e_embd)
        outputs = self.output_MHA(concat_attention)
        outputs = self.dropout(outputs)
        # (bs, n_q_seq, hid_dim), (bs, n_heads, n_q_seq, n_k_seq)
        return outputs

""" feed forward """
class PositionwiseFeedforwardLayer(nn.Module):
    def __init__(self):
        super(PositionwiseFeedforwardLayer, self).__init__()
        self.linear_1 = nn.Linear(hid_dim, pf_dim)
        self.linear_2 = nn.Linear(pf_dim, hid_dim)

    def forward(self, attention):
        output = self.linear_1(attention)
        output = F.relu(output)
        output = self.linear_2(output)
        return output

class EncoderLayer(nn.Module):
    def __init__(self, hid_dim=768, n_heads=12, pf_dim=3072, dropout_prob=0.1):
        super(EncoderLayer, self).__init__()
        
        self.attn = MultiHeadAttentionLayer( )
        self.ffn = PositionwiseFeedforwardLayer( )
        
        self.layernorm1 = nn.LayerNorm(hid_dim)
        self.layernorm2 = nn.LayerNorm(hid_dim)
        
        self.dropout1 = nn.Dropout(p=dropout_prob)
        self.dropout2 = nn.Dropout(p=dropout_prob)
        self.dropout3 = nn.Dropout(p=dropout_prob)
        
    def forward(self, x):
        # apply dropout
        out_1 = self.dropout1(x)
        # apply layer normalization
        out_2 = self.layernorm1(out_1)
        # compute multi-head self-attention
        msa_out = self.attn(out_2, out_2, out_2)
        # apply dropout
        out_3 = self.dropout2(msa_out)
        # apply residual connection
        res_out = x + out_3
        # apply layer normalization
        out_4 = self.layernorm2(res_out)
        # compute mlp output
        mlp_out = self.ffn(out_4)
        # apply dropout
        out_5 = self.dropout3(mlp_out)
        # apply residual connection
        ffn_outputs = res_out + out_5
        
        return ffn_outputs

# unit test on transformer encoder

# dropout probability
dropout_prob = 0.1
n_heads = 8
# init transformer encoder
transformer_encoder = EncoderLayer(D, n_heads, pf_dim, dropout_prob)

# compute transformer encoder output
output = transformer_encoder(patch_embeddings)

assert output.shape == (B, N + 1, D)
output.shape

class MLPHead(nn.Module):
    def __init__(self, hid_dim=768, num_classes=10, fine_tune=False):
        super(MLPHead, self).__init__()
        self.num_classes = num_classes
        
        if not fine_tune:
            # hidden layer with tanh activation function 
            self.mlp_head = nn.Sequential(
                                    nn.Linear(hid_dim, 3072),  # hidden layer
                                    nn.Tanh(),
                                    nn.Linear(3072, num_classes)    # output layer
                            )
        else:
            # single linear layer
            self.mlp_head = nn.Linear(hid_dim, num_classes)
        
    def forward(self, x):
        x = self.mlp_head(x)
        return x
        
# unit test on classification head


# extract [class] token from transformer encoder output
z_L = output[0][0]   # extract batch and [class] token  

# init number of classes
n_class = 10

# init classification head for pre-training phase
mlp_head_pretrain = MLPHead(D, n_class)
# init classification head for fine-tuning phase
mlp_head_finetune = MLPHead(D, n_class, fine_tune=True)

# compute mlp head output
output_1 = mlp_head_pretrain(z_L)
# compute mlp head output for fine-tuning 
output_2 = mlp_head_finetune(z_L)

# size of output
print(output_1.size(dim=0))

# assert output is consistent with number of classes
assert output_1.size(dim=0) == n_class
assert output_2.size(dim=0) == n_class

class VisionTransformer(nn.Module):
    def __init__(self, patch_size=16, image_size=224, channel_size=3, 
                     num_layers=12, hid_dim=384, n_heads=12, pf_dim=1024, 
                            dropout_prob=0.1, num_classes=10, pretrain=True):
        super(VisionTransformer, self).__init__()
        
        self.patch_size = patch_size 
        self.channel_size = channel_size 
        self.num_layers = num_layers
        self.hid_dim = hid_dim
        self.n_heads = n_heads
        self.pf_dim = pf_dim
        self.dropout_prob = dropout_prob
        self.num_classes = num_classes
        
        # get number of patches of the image
        self.num_patches = int(image_size ** 2 / patch_size ** 2)   # height * width / patch size ^ 2   
        
        # trainable linear projection for mapping dimnesion of patches (weight matrix E)
        self.W = nn.Parameter(
                    torch.randn( patch_size * patch_size * channel_size, hid_dim))
        
        # position embeddings (E_pos)
        self.pos_embedding = nn.Parameter(torch.randn(self.num_patches + 1, hid_dim))
        
        # learnable class token embedding (x_class)
        self.class_token = nn.Parameter(torch.rand(1, D))
        
        # stack transformer encoder layers 
        transformer_encoder_list = [
            EncoderLayer(hid_dim, n_heads, pf_dim, dropout_prob) 
                    for _ in range(num_layers)] 
        self.transformer_encoder_layers = nn.Sequential(*transformer_encoder_list)
        
        # mlp head
        self.mlp_head = MLPHead(hid_dim, num_classes)
        
    def forward(self, x):
        # get patch size and channel size
        P, C = self.patch_size, self.channel_size
        
        # split image into patches
        patches = x.unfold(1, C, C).unfold(2, P, P).unfold(3, P, P)
        patches = patches.contiguous().view(patches.size(0), -1, C * P * P).float()
        
        # linearly embed patches
        patch_embeddings = torch.matmul(patches , self.W)
        
        # add class token
        batch_size = patch_embeddings.shape[0]
        patch_embeddings = torch.cat((self.class_token.repeat(batch_size, 1, 1), patch_embeddings), 1)
        
        # add positional embedding
        patch_embeddings = patch_embeddings + self.pos_embedding
        
        # feed patch embeddings into a stack of Transformer encoders
        transformer_encoder_output = self.transformer_encoder_layers(patch_embeddings)
        
        # extract [class] token from encoder output
        output_class_token = transformer_encoder_output[:, 0]
        
        # pass token through mlp head for classification
        final_output = self.mlp_head(output_class_token)
        return final_output

# unit test on vision transformer

image_size = 224; channel_size = 3

# number of classes CIFAR-10
n_class = 10

# dropout probability
dropout_prob = 0.1

# read and resize image 
image = Image.open('car.png').resize((image_size, image_size))

# convert PIL image to tensor
X = T.PILToTensor()(image)

# add batch dimension 
X =  X[None, ...]    # batch size = 1

# An Image Is Worth 16x16 Words
patch_size = 16

# init vision transformer model
vision_transformer = VisionTransformer(patch_size, image_size, channel_size, 
                            n_layer, hid_dim, n_heads, pf_dim, dropout_prob, n_class)

# compute vision transformer output
vit_output = vision_transformer(X)

assert vit_output.size(dim=1) == n_class
print(vit_output.shape)

# get class probabilities
probabilities = F.softmax(vit_output[0], dim=0)

# probabilities should sum up to 1
print(torch.sum(probabilities))

import torchvision
from torchvision import datasets

from torch.utils.data import DataLoader
from torch.utils.data.sampler import SubsetRandomSampler

import torchvision.models as models

import torch.optim as optim
import torch.optim.lr_scheduler as lr_scheduler

# Load data 
#
# We will use a subset of CIFAR10 dataset

image_size = 224

# define transform
transform = T.Compose([ T.Resize(image_size), T.ToTensor() ])

torchvision.datasets.CIFAR10.url="http://www.cs.toronto.edu/~kriz/cifar-10-python.tar.gz"

# init CIFAR10 training and test datasets 
trainset = datasets.CIFAR10(root='data', train=True, download=True, transform=transform)
testset = datasets.CIFAR10(root='data', train=False, download=True, transform=transform)

# get class names
classes = trainset.classes

# get a subset of the trainset and test set
trainset = torch.utils.data.Subset(trainset, list(range(5000)))
testset = torch.utils.data.Subset(testset, list(range(1000)))

# output classes
classes

# define data loaders

batch_size = 128

# percentage of training set to use as validation
valid_size = 0.2

# get training indices that wil be used for validation
train_size = len(trainset)
indices = list(range(train_size))
np.random.shuffle(indices)
split = int(np.floor(valid_size * train_size))
train_idx, valid_idx = indices[split:], indices[:split]

# define samplers to obtain training and validation batches
train_sampler = SubsetRandomSampler(train_idx)
valid_sampler = SubsetRandomSampler(valid_idx)

# prepare data loaders
train_loader = DataLoader(trainset, batch_size=batch_size, sampler=train_sampler)
valid_loader = DataLoader(trainset, batch_size=batch_size, sampler=valid_sampler)
test_loader = DataLoader(testset, batch_size=batch_size)

# print out classes statistics

# get all training samples labels
train_labels = [labels for i, (images, labels) in enumerate(train_loader)]
train_labels = torch.cat((train_labels), 0)
train_labels_count = train_labels.unique(return_counts=True)

# print(train_labels_count)

print('The number of samples per classes in training dataset:\n')
for label, count in zip(train_labels_count[0], train_labels_count[1]):
    print('\t {}: {}'.format(label, count))

# get all test samples labels
test_labels = [labels for i, (images, labels) in enumerate(test_loader)]
test_labels = torch.cat((test_labels), 0)
test_labels_count = test_labels.unique(return_counts=True)

print()
print('The number of samples per classes in test dataset:\n')
for label, count in zip(test_labels_count[0], test_labels_count[1]):
    print('\t {}: {}'.format(label, count))

# define model

vision_transformer = models.vit_b_16(weights=models.ViT_B_16_Weights.DEFAULT)

vision_transformer

# vit-16 model is trained on ImageNet 
# we expect to have output of 1000 number of classes

vision_transformer.heads

# fine-tune with dataset

# change the number of output classes
vision_transformer.heads = nn.Linear(in_features=768, out_features=len(classes), bias=True)

# freeze the parameters except the last linear layer
#
# freeze weights
for p in vision_transformer.parameters():
    p.requires_grad = False

# unfreeze weights of classification head to train
for p in vision_transformer.heads.parameters():
    p.requires_grad = True

# check whether corresponding layers are frozen

for layer_name, p in vision_transformer.named_parameters():
    print('Layer Name: {}, Frozen: {}'.format(layer_name, not p.requires_grad))
    print()

# specify loss function
criterion = nn.CrossEntropyLoss()

# define optimizer
# only train the parameters with requires_grad set to True
optimizer = optim.Adam(filter(lambda p: p.requires_grad, vision_transformer.parameters()), lr=0.0001)

# Check for a GPU
train_on_gpu = torch.cuda.is_available()
train_on_gpu

# load model if it exists
import os

model_path = 'vit.pth'

if os.path.exists(model_path):
    vision_transformer.load_state_dict(torch.load(model_path))

# Train model 

# number of epochs
n_epoch = 5

# number of iterations to save model
n_step=100

train_loss_list, valid_loss_list = [], []

# move model to GPU
if train_on_gpu:
    vision_transformer.to('cuda')

# prepare model for training
vision_transformer.train()


from tqdm import tqdm, tqdm_notebook, trange

for epoch in range(n_epoch):

    with tqdm_notebook(total=len(train_loader), desc=f"Train Epoch {epoch+1}") as pbar:
        train_loss = 0.0
        valid_loss = 0.0

        # get batch data
        for i, (images, targets) in enumerate(train_loader):

            # move to gpu if available
            if train_on_gpu:
                images, targets = images.to('cuda'), targets.to('cuda')

            # clear grad
            optimizer.zero_grad()

            # feedforward data
            outputs = vision_transformer(images)

            # calculate loss
            loss = criterion(outputs, targets)

            # backward pass, calculate gradients
            loss.backward()

            # update weights
            optimizer.step()

            # track loss
            train_loss += loss.item()

            # save the model parameters
            if i % n_step == 0:
                torch.save(vision_transformer.state_dict(), model_path)
            pbar.update(1)
        # set model to evaluation mode
        vision_transformer.eval()

        # validate model
        for images, targets in valid_loader:

            # move to gpu if available
            if train_on_gpu:
                images = images.to('cuda')
                targets = targets.to('cuda')

            # turn off gradients
            with torch.no_grad():

                outputs = vision_transformer(images)
                loss = criterion(outputs, targets)
                valid_loss += loss.item()

        # set model back to trianing mode
        vision_transformer.train()

        # get average loss values
        train_loss = train_loss / len(train_loader)
        valid_loss = valid_loss / len(valid_loader)

        train_loss_list.append(train_loss)
        valid_loss_list.append(valid_loss)

        # output training statistics for epoch
        print('Epoch: {} \t Training Loss: {:.6f} \t Validation Loss: {:.6f}'
                      .format( (epoch+1), train_loss, valid_loss))
    
# visualize loss statistics

plt.title("Training and Validation Loss")
plt.xlabel("Epoch")
plt.ylabel("Loss")

# plot losses
x = list(range(1, n_epoch + 1))
plt.plot(x, train_loss_list, color ="blue", label='Train')
plt.plot(x, valid_loss_list, color="orange", label='Validation')
plt.legend(loc="upper right")
plt.xticks(x)

plt.show()

# prepare model for evaluation
vision_transformer.eval()

test_loss = 0.0
accuracy = 0

# number of classes
n_class = len(classes)

class_correct = np.zeros(n_class)
class_total = np.zeros(n_class)

# move model back to cpu
vision_transformer = vision_transformer.to('cpu')

# test model
for images, targets in test_loader:
    
    # get outputs
    outputs = vision_transformer(images)
    
    # calculate loss
    loss = criterion(outputs, targets)
    
    # track loss
    test_loss += loss.item()
    
    # get predictions from probabilities
    preds = torch.argmax(F.softmax(outputs, dim=1), dim=1)
    
    # get correct predictions
    correct_preds = (preds == targets).type(torch.FloatTensor)
    
    # calculate and accumulate accuracy
    accuracy += torch.mean(correct_preds).item() * 100
    
    # calculate test accuracy for each class
    for c in range(n_class):
        
        class_total[c] += (targets == c).sum()
        class_correct[c] += ((correct_preds) * (targets == c)).sum()

# get average accuracy
accuracy = accuracy / len(test_loader)

# get average loss 
test_loss = test_loss / len(test_loader)

# output test loss statistics 
print('Test Loss: {:.6f}'.format(test_loss))
    
class_accuracy = class_correct / class_total

print('Test Accuracy of Classes')
print()

for c in range(n_class):
    print('{}\t: {}% \t ({}/{})'.format(classes[c], 
                                int(class_accuracy[c] * 100), int(class_correct[c]), int(class_total[c])) )

print()
print('Test Accuracy of Dataset: \t {}% \t ({}/{})'.format(int(accuracy), 
                                                           int(np.sum(class_correct)), int(np.sum(class_total)) ))





# Neural nets with PyTorch
# small version of nets used in MuZero paper

import torch
import torch.nn as nn
import torch.nn.functional as F
import numpy as np

num_filters = 256
num_blocks = 16

def get_action_feature(action):
    # input tensor for neural nets (action)
    a = np.zeros((1, 19, 19), dtype=np.float32)
    if action<361 :
        a[0, action // 19, action % 19] = 1
    return a

class Conv(nn.Module):
    def __init__(self, filters0, filters1, kernel_size, bn=False):
        super().__init__()
        self.conv = nn.Conv2d(filters0, filters1, kernel_size, stride=1, padding=kernel_size//2, bias=False)
        self.bn = None
        if bn:
            self.bn = nn.BatchNorm2d(filters1)

    def forward(self, x):
        h = self.conv(x)
        if self.bn is not None:
            h = self.bn(h)
        return h

class ResidualBlock(nn.Module):
    def __init__(self, filters):
        super().__init__()
        self.conv = Conv(filters, filters, 3, True)

    def forward(self, x):
        y = F.relu(self.conv(x))
        y = F.relu(x+self.conv(y))
        return y


class Representation(nn.Module):
    ''' Conversion from observation to inner abstract state '''
    def __init__(self, input_shape):
        super().__init__()
        self.input_shape = input_shape
        self.board_size = self.input_shape[1] * self.input_shape[2]

        self.layer0 = Conv(self.input_shape[0], num_filters, 3, bn=True)
        self.blocks = nn.ModuleList([ResidualBlock(num_filters) for _ in range(num_blocks)])

    def forward(self, x):
        h = F.relu(self.layer0(x))
        for block in self.blocks:
            h = block(h)
        return h

    def inference(self, x):
        self.eval()
        with torch.no_grad():
            rp = self(torch.from_numpy(x).unsqueeze(0))
        return rp.cpu().numpy()[0]

class Prediction(nn.Module):
    ''' Policy and value prediction from inner abstract state '''
    def __init__(self, action_shape):
        super().__init__()
        self.board_size = np.prod(action_shape[1:])
        self.action_size = action_shape[0] * self.board_size + 1

        self.conv_p = Conv(num_filters, 2, 1, bn=True)
        self.fc_p = nn.Linear(self.board_size*2, 362, bias=True)

        self.conv_v = Conv(num_filters, 1, 1, bn=True)
        self.fc_v1 = nn.Linear(self.board_size, 256, bias=True)
        self.fc_v2 = nn.Linear(256, 1, bias=True)
        

    def forward(self, rp):
        h_p = F.relu(self.conv_p(rp))
        h_p = self.fc_p(h_p.view(-1, self.action_size))

        h_v = F.relu(self.conv_v(rp))
        h_v = self.fc_v1(h_v.view(-1, 256))
        h_v = self.fc_v2(h_v.view(256, 1))

        # range of value is -1 ~ 1
        return F.softmax(h_p, dim=-1), torch.tanh(h_v)

    def inference(self, rp):
        self.eval()
        with torch.no_grad():
            p, v = self(torch.from_numpy(rp).unsqueeze(0))
        return p.cpu().numpy()[0], v.cpu().numpy()[0][0]

class Dynamics(nn.Module):
    '''Abstruct state transition'''
    def __init__(self, rp_shape, act_shape):
        super().__init__()
        self.rp_shape = rp_shape
        self.layer0 = Conv(rp_shape[0] + act_shape[0], num_filters, 3, bn=True)
        self.blocks = nn.ModuleList([ResidualBlock(num_filters) for _ in range(num_blocks)])

    def forward(self, rp, a):
        h = torch.cat([rp, a], dim=1)
        h = F.relu(self.layer0(h))
        for block in self.blocks:
            h = block(h)
        return h

    def inference(self, rp, a):
        self.eval()
        with torch.no_grad():
            rp = self(torch.from_numpy(rp).unsqueeze(0), torch.from_numpy(a).unsqueeze(0))
        return rp.cpu().numpy()[0]

class Nets(nn.Module):
    '''Whole nets'''
    def __init__(self):
        super().__init__()
        input_shape = (18, 19, 19)
        action_shape = (1, 19, 19)
        rp_shape = (num_filters, *input_shape[1:])

        self.representation = Representation(input_shape)
        self.prediction = Prediction(action_shape)
        self.dynamics = Dynamics(rp_shape, action_shape)

    def predict_all(self, state0, path):
        '''Predict p and v from original state and path'''
        outputs = []
        self.eval()
        x = torch.from_numpy(state0).unsqueeze(0)
        with torch.no_grad():
            rp = self.representation(x)
            outputs.append(self.prediction(rp))
            for action in path:
                a = get_action_feature(action).unsqueeze(0)
                rp = self.dynamics(rp, a)
                outputs.append(self.prediction(rp))
        #  return as numpy arrays
        return [(p.cpu().numpy()[0], v.cpu().numpy()[0][0]) for p, v in outputs]


# Training of neural nets

import torch.optim as optim


batch_size = 256
num_epochs = 10
num_steps = 8000
FK = 5
def gen_target(games):
    '''Generate inputs and targets for training'''
    # path, reward, observation, action, policy
    batch_idx = np.random.randint(0, len(games), batch_size)
    batchdata = []
    for i in range(batch_size): 
        game = games[batch_idx[i]]
        s0 = np.random.randint(len(game)-FK)
        state0 = np.zeros(18, 361).astype(np.uint8)
        actions = np.zeros(FK, 362).astype(np.int16)
        probs = np.zeros(FK+1, 362).astype(np.float32)
        winners = np.zeros(FK+1, 1).astype(np.float32)
        for j in range(8) :
            k = s0 - j
            if k < 0:
                k=0
            state0[2*j][0:361] = game[k][0]
            state0[2*j+1][0:361] = game[k][1]
            stm = game[s0][2]
            if stm == 0:  #black to move
                state0[16][0:361] = 1
            else :
                state0[17][0:361] = 1
        for j in range(FK+1) :
            probs[j][0:362] = game[s0+j][4]
            winners[j][0]   = game[s0+j][5]
            if j < FK :
                action = game[s0+j][3]
                actions[j][action] = 1
        batchdata.append((state0, actions, probs, winners))
        
    return batchdata



def train(games, nets=Nets()):
    '''Train neural nets'''
    optimizer = optim.SGD(nets.parameters(), lr=1e-3, weight_decay=1e-4, momentum=0.75)
    for epoch in range(num_epochs):
        p_loss_sum, v_loss_sum = 0, 0
        nets.train()
        for i in range(num_steps):
            k = 4#np.random.randint(4)
            x, ax, p_target, v_target = zip(*[gen_target(episodes[np.random.randint(len(episodes))], k) for j in range(batch_size)])
            x = torch.from_numpy(np.array(x))
            ax = torch.from_numpy(np.array(ax))
            p_target = torch.from_numpy(np.array(p_target))
            v_target = torch.FloatTensor(np.array(v_target))
            
            # Change the order of axis as [time step, batch, ...]
            ax = torch.transpose(ax, 0, 1)
            p_target = torch.transpose(p_target, 0, 1)
            v_target = torch.transpose(v_target, 0, 1)

            p_loss, v_loss = 0, 0

            # Compute losses for k (+ current) steps
            for t in range(k + 1):
                rp = nets.representation(x) if t == 0 else nets.dynamics(rp, ax[t - 1])
                p, v = nets.prediction(rp)
                p_loss += torch.sum(-p_target[t] * torch.log(p))
                v_loss += torch.sum((v_target[t] - v) ** 2)

            p_loss_sum += p_loss.item()
            v_loss_sum += v_loss.item()

            optimizer.zero_grad()
            (p_loss + v_loss).backward()
            optimizer.step()

        for param_group in optimizer.param_groups:
            param_group['lr'] *= 0.85
    print('p_loss %f v_loss %f' % (p_loss_sum / len(episodes), v_loss_sum / len(episodes)))
    return nets

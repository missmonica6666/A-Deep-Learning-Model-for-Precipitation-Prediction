
from code import MetNet
import torch
import numpy as np
from torch import nn, optim
import time
import math


cuda       = True
ngpu       = 2
connect = True

# seed = 0
# torch.manual_seed(seed)
# np.random.seed(seed)

if cuda == True:
    torch.cuda.set_device('cuda:0,1')

train_x = torch.load("./code/lasttry/x.pth")
train_y = torch.load("./code/lasttry/y.pth")
num = train_x.shape[0]
# train_x = train_x.resize_(num,4,20,96,96)
# train_y = train_y.resize_(num,4,20,96,96)
train_x = train_x.cuda()
train_y = train_y.cuda()

T = 20
batch_size = 2
criterion = nn.MSELoss()

def random_choice():
    total = train_x.shape[0]

    
    X = []
    Y = []
    for _ in range(batch_size):
        choose = np.random.randint(0, total-1)
        # print('choose:',choose)
        video_train = train_x[choose]
        video_real = train_y[choose]
        X.append(video_train)
        Y.append(video_real)

        
    X = torch.stack(X,dim = 0)
    Y = torch.stack(Y,dim = 0)
    
    return X,Y



flag = 0
if connect:
    net = MetNet(ngpu=ngpu)
    checkpoint = torch.load("./code/lasttry/trained_models/data500/epoch2101.pth")   
    net.load_state_dict(checkpoint['net'])  # 加载模型可学习参数
    flag = 1




else:
    net = MetNet(ngpu=ngpu)

optimizer = torch.optim.Adadelta(net.parameters(), lr=1, rho=0.9, eps=1e-08, weight_decay=0)

if flag ==1:
    optimizer.load_state_dict(checkpoint['optimizer'])
    for state in optimizer.state.values():
        for k, v in state.items():
            if torch.is_tensor(v):
                state[k] = v.cuda()  # 加载优化器参数



def timeSince(since):
    now = time.time()
    s = now - since
    d = math.floor(s / ((60**2)*24))
    h = math.floor(s / (60**2)) - d*24
    m = math.floor(s / 60) - h*60 - d*24*60
    s = s - m*60 - h*(60**2) - d*24*(60**2)
    return '%dd %dh %dm %ds' % (d, h, m, s)

if cuda == True:
    net.cuda()
   
    criterion.cuda()


    
def train(epochs, model):
  
    # optimizer = torch.optim.Adadelta(model.parameters(), lr=1.0, rho=0.9, eps=1e-06, weight_decay=0)
    

    
    start_time = time.time()
    for e in range(epochs):
        batch_x,batch_y = random_choice()
        
        batch_x = batch_x.float()
        batch_y = batch_y.float()

      
        out = model(batch_x)
        loss = criterion(out, batch_y)
           
        optimizer.zero_grad() # if don't call zero_grad, the grad of each batch will be accumulated
        loss.backward()
        optimizer.step()

        checkpoint = {
        "net": model.state_dict(),
        'optimizer':optimizer.state_dict(),
        "epoch": epochs
        }

        if e%20 == 0:
            print('[%d/%d] (%s) Loss: %.4f '
                % (e, epochs, timeSince(start_time), loss.data))

                
        if e %300 == 0:
            torch.save(checkpoint, './code/lasttry/trained_models/data500/epoch'+str(e+1)+'.pth')

    
        




      
  
train(epochs = 200000, model = net)
    
 
        


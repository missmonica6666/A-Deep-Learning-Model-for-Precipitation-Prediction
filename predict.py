
from code import MetNet
import torch
import torch.nn as nn
from PIL import Image

import torch.utils.data as Data
import os
from torchvision.utils import make_grid

''' prepare dataset '''

train_x = torch.load("./code/mygan/x.pth")
train_y = torch.load("./code/mygan/y.pth")
num = train_x.shape[0]
# train_x = train_x.resize_(num,4,20,96,96)
# train_y = train_y.resize_(num,4,20,96,96)
# train_x = train_x[0].cuda()
# train_y = train_y[0].cuda()
# train_y = train_y.resize_(4,20,96,96)
print(train_y.shape)

torch.cuda.set_device('cuda:0,1')

# def choose(chose,batch_size=1):
#     total = train_x.shape[0]

    
#     X = []
#     Y = []
#     if chose:
#         video_train = train_x[chose]
#         video_real = train_y[chose]
#         print("video_train:",video_train.shape)
#         X.append(video_train)
#         Y.append(video_real)

#     else:
#         for _ in range(batch_size):
#             choose = np.random.randint(0, total-1)
#             video_train = train_x[choose]
#             video_real = train_y[choose]
            # X.append(video_train)
            # Y.append(video_real)
            # print(choose)
       
    # X = torch.stack(X,dim = 0)
    # Y = torch.stack(Y,dim = 0)
    
    # return video_train,video_real

def predict(chose,net,device):
    global train_x
    global train_y

    net.cuda()

    # batch_x,batch_y = choose(chose)
    
    Y = [train_y[chose]]
    batch_y = torch.stack(Y,dim = 0)
    # batch_y = train_y.resize_(1,4,20,96,96)

    X = [train_x[chose]]
    batch_x = torch.stack(X,dim = 0)
    # batch_x = train_x.resize_(1,4,20,96,96)


    batch_size = 1
    T = 20

        
    for i in range(1):
        with torch.no_grad():
            batch_x.cuda()
            batch_y.cuda()
  
            print("batch_x:",batch_x.shape)
            batch_x = batch_x.float()
            batch_y = batch_y.float()
            fake_videos = net(batch_x)
            print('fake_videos:',fake_videos.shape)

            predictt = fake_videos.detach()
            predictt = predictt.cpu().numpy()
                 
            predictt = predictt.transpose(0,2,1,3,4)
            
            print(predictt.shape)
            batch_y = batch_y.detach()
        for ii in range(20):
            for j in range(4):
                im = Image.fromarray(predictt[0][ii][j]*255)
                im = im.convert('L')
                im.save('./code/lasttry/predict/9_7/choose'+str(chose)+'_'+str(ii)+'_'+str(j)+'.jpg')
            
                true = batch_y.cpu().numpy().transpose(0,2,1,3,4)
                im0 = Image.fromarray(true[0][ii][j]*255)
                im0 = im0.convert('L')
                im0.save('./code/lasttry/predict/9_7/true'+str(ii)+'_'+str(j)+'.jpg')
            



device = torch.device('cuda:0' if torch.cuda.is_available() else 'cpu')



model = MetNet(ngpu=2)
checkpoint = torch.load("./code/lasttry/trained_models/data500/epoch2101.pth")   
model.load_state_dict(checkpoint['net']) 


predict(0,model,device)



# batch_x,true = choose([172])
# train_y = true[0].resize_(4,20,360,360)
# train_y = train_y[172]
# print("train_y,raw:",train_y.shape)
# train_y = train_y.resize_(4,20,96,96)
# print("train_y,small:",train_y.shape)
# train_y = train_y.resize_(4,20,360,360)
# print("train_y,resize:",train_y.shape)


# print("train_y,raw:",train_y.shape)
# Y = [train_y[172]]
# train_y = torch.stack(Y,dim = 0)
# train_y = train_y.resize_(1,4,20,96,96)
# print("train_y,small:",train_y.shape)
# train_y = train_y.resize_(1,4,20,360,360)
# print("train_y,resize:",train_y.shape)
# train_y = train_y[172]

# # print("train_y,raw:",train_y.shape)
# # train_y = train_y.resize_(num,4,20,96,96)
# # print("train_y,small:",train_y.shape)

# # Y = [train_y[172]]
# # train_y = torch.stack(Y,dim = 0)
# # print(train_y.shape)
# # train_y = train_y.resize_(1,4,20,360,360)
# # print("train_y,resize:",train_y.shape)
# # train_y = train_y[0]





# for i in range(20):
#     for j in range(4):
#         true = train_y[0].cpu().numpy().transpose(1,0,2,3)
#         im0 = Image.fromarray(true[i][j]*255)
#         im0 = im0.convert('L')
#         im0.save('./code/lasttry/predict/9_6/trueeemost1'+str(i)+'_'+str(j)+'.jpg')





from turtle import forward
from xml.etree.ElementTree import tostring
import gym
import random
import torch
import torch.nn as TNN
import torch.nn.functional as TF
import numpy as np
from collections import deque
import torch.utils.data as Data
import matplotlib.pyplot as plt
from functools import partial

ENV_NAME = 'CartPole-v1'
EPISODE = 100000
BUFFER_SIZE = 1000000
INITIAL_EPSILON=1
FINAL_EPSILON=0.1
GAMMA = 0.99
BATCHSIZE = 32
STEP = 1000
TEST = 10
SAVINGPATH = "./modelDQN_NATURE/"
CSTEP = 10

# ReplayBuffer
class ReplayBuffer():
    def __init__(self,env,buffersize):
        self.buffer = deque(maxlen=buffersize)
        self.state_dim = env.observation_space.shape[0]
        self.action_dim = env.action_space.n 
    def append(self,content):
        onehot_action = np.zeros(self.action_dim)
        onehot_action[content[1]]=1
        self.buffer.append([content[0],onehot_action,content[2],content[3],content[4]])

# 生成标签 y 及 minibatch
def sample_Batch(replaybuffer,Q_t,batch_size=BATCHSIZE):
    minibatch = random.sample(replaybuffer.buffer,BATCHSIZE)
    
    state_batch = [data[0] for data in minibatch]
    action_batch = [data[1] for data in minibatch]
    reward_batch = [data[2] for data in minibatch]
    next_state_batch = [data[4] for data in minibatch]

    tensor_state = torch.Tensor(state_batch).reshape(BATCHSIZE,Q_t.state_dim)
    tensor_action = torch.Tensor(action_batch).reshape(BATCHSIZE,Q_t.action_dim)
    
    # 生成标签y
    y_batch=[each[2] if each[3] else (reward_batch[i] + GAMMA*torch.max(Q_t(torch.Tensor(next_state_batch[i]))).detach().numpy()) for i,each in enumerate(minibatch)]
    tensor_y = torch.Tensor(y_batch)

    return tensor_y,tensor_action,tensor_state

#Network
class DQN(TNN.Module):
    def __init__(self,env,target=False) :
        super(DQN,self).__init__()
        self.state_dim = env.observation_space.shape[0]
        self.action_dim = env.action_space.n 
        self.net = TNN.Sequential(
            TNN.Linear(self.state_dim,20),
            TNN.ReLU(),
            TNN.Linear(20,20),
            TNN.ReLU(),
            TNN.Linear(20,self.action_dim),
        )
        if target: # target 网络，进行初始化，不是设置优化器
            # 用偏函数定义初始化函数
            init = partial(TNN.init.uniform_,a=0.0,b=0.0) 
            for layer in self.net:
                # 仅对 Linear 层初始化
                if isinstance(layer,torch.nn.modules.linear.Linear):
                    init(layer.weight)
        else:
            self.epsilon = INITIAL_EPSILON
            self.train_setup()

    def forward(self,x):
        Q_value = self.net(x)
        return Q_value

    def action(self,state):
        return np.argmax(self.net(torch.tensor(state)).detach().numpy())

    def egreedy_action(self,state):
        self.epsilon = max(FINAL_EPSILON,self.epsilon - INITIAL_EPSILON/10000)
        if(random.random()<self.epsilon): return random.randint(0,self.action_dim-1) 
        else: return np.argmax(self.net(torch.tensor(state)).detach().numpy())
        
    def load_model(self,model_dict):
        self.net.load_state_dict(model_dict)
    
    # 训练初始化
    def train_setup(self):
        self.optimizer = torch.optim.RMSprop(self.net.parameters(),lr=0.00025,alpha=0.95,momentum=0.95)
        self.loss_fn = TNN.MSELoss()

def CartPole_agent_reward(next_state,done,i):
    if not done:
        reward_agent =  0.2-abs(next_state[0])*0.1#0.1#0.3-np.abs(next_state[0])*0.1
    elif i  == 999:
        reward_agent =  0.2-abs(next_state[0])*0.1
    else:
        reward_agent= -1
    return reward_agent

# 更新 Q_t
def update_Q_t(Q,Q_t):
    Q_t.load_state_dict(Q.state_dict())

# 更新 Q
def update_Q(replaybuffer,Q,Q_t):
    # 经验池大小未达到batch大小
    if len(replaybuffer.buffer)<BATCHSIZE:
        return
    # 用target网络产生y作为更新Q网络的标签
    tensor_y,tensor_action,tensor_state = sample_Batch(replaybuffer,Q_t)

    pred = Q(tensor_state)  #计算DQN对本step的Q_value估计
    Q_action = torch.sum(pred*tensor_action,dim=1)#保留本step采取action对应的Qvalue（DQN预测的）
    loss = Q.loss_fn(tensor_y,Q_action)

    # 反向传播
    Q.optimizer.zero_grad()
    loss.backward()
    Q.optimizer.step()

def main():
    env = gym.make(ENV_NAME)
    replaybuffer = ReplayBuffer(env,BUFFER_SIZE)
    Q = DQN(env) # 动作网络
    Q_t = DQN(env,target=True) #目标网络，产生y
    Test_rec = [] # 
    for episode in range(EPISODE):
        state = env.reset()
        for i in range(1,STEP):
            # 使用Q产生动作
            action = Q.egreedy_action(state)
            next_state,reward,done,_ = env.step(action=action)

            # 取reward
            reward_agent = CartPole_agent_reward(next_state,done,i)

            # 将观测数据存入经验池
            replaybuffer.append([state,action,reward_agent,done,next_state])

            # 更新Q网络
            update_Q(replaybuffer,Q,Q_t)

            # 维护下一轮状态
            state = next_state

            # 每C轮，更新一次target网络
            if i % CSTEP == 0:
                update_Q_t(Q,Q_t)

            if done: #对于环境最大step，直接放弃该数据
                break
        # 每10个episode测试一下
        if episode % 10==0:
            # 初始化测试参数
            total_reward, max_a_reward, min_a_reward = 0, 0, 1000
            for j in range(TEST): #test 10 
                state = env.reset()
                acc_reward = 0
                for i in range(1000):
                    action = Q.action(state)
                    state,reward,done,_ = env.step(action=action)
                    acc_reward+=reward
                    if done :
                        break

                # 记录回报
                total_reward += acc_reward
                # 求本次最大值和最小值
                max_a_reward = max(max_a_reward,acc_reward)
                min_a_reward = min(min_a_reward,acc_reward)

            # 将结果记录
            Test_rec.append([min_a_reward,total_reward/TEST,max_a_reward])
            print(f'EPISODE {episode}  acc_reward: {acc_reward}   max_reward: {max_a_reward}   mean_reward: {Test_rec[-1]}')
            # 保存每次测试都达到1000step的网络参数
            if (total_reward/TEST) == 1000: 
                fileName = 'optim-model-episode'+str(total_reward/TEST)+'.pth'
                torch.save(Q.net.state_dict(), SAVINGPATH + fileName)
                break
            
if __name__ == '__main__':
    main()

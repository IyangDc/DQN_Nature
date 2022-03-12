# DQN_Nature
DQN_Nature version coded with pytorch

加入了target网络的DQN比较稳定，收敛快速。

使用论文中推荐的RMSprop优化器效果优于Adam优化器

此外本项目并未按照论文中要求在训练开始前随机取Action生成数据填充经验池
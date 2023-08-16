import numpy as np#求最大值的索引等会用到
import time#导入时间模块
import torch#保存和导入Q表时用到
import tkinter as tk#图行界面化设计
import pandas as pd#创建Q表时用到
import matplotlib.pyplot as plt#画奖励值图像用到
class Maze(tk.Tk):#创建地图
    UNIT = 50  # 像素大小
    MAZE_H = 9  # 画布长度
    MAZE_W = 9  # 画布宽度
    states=81#共有81种状态(0~80)
    actions=4#共有4种动作(0,1,2,3)上下左右
    def __init__(self):#设置地图
        super().__init__()
        self.title('迷宫Q_earning')
        h = self.MAZE_H * self.UNIT#窗口的高
        w = self.MAZE_W * self.UNIT#窗口的宽
        self.canvas = tk.Canvas(self, bg='pink', height=h, width=w)#窗口的颜色设置
        # 画出网格的线条
        for c in range(0, w, self.UNIT):
            self.canvas.create_line(c, 0, c, h)
        for r in range(0, h, self.UNIT):
            self.canvas.create_line(0, r, w, r)
        # 画出障碍物，并设置黑色
        self._draw_rect(1, 1, 'grey47')
        self._draw_rect(3, 0, 'grey47')
        self._draw_rect(6, 0, 'grey47')
        self._draw_rect(5, 1, 'grey47')
        self._draw_rect(3, 2, 'grey47')
        self._draw_rect(8, 2, 'grey47')
        self._draw_rect(0, 3, 'grey47')
        self._draw_rect(1, 3, 'grey47')
        self._draw_rect(3, 3, 'grey47')
        self._draw_rect(4, 3, 'grey47')
        self._draw_rect(5, 3, 'grey47')
        self._draw_rect(7, 3, 'grey47')
        self._draw_rect(1, 5, 'grey47')
        self._draw_rect(3, 5, 'grey47')
        self._draw_rect(6, 5, 'grey47')
        self._draw_rect(7, 5, 'grey47')
        self._draw_rect(2, 6, 'grey47')
        self._draw_rect(8, 4, 'grey47')
        self._draw_rect(4, 6, 'grey47')
        self._draw_rect(5, 7, 'grey47')
        self._draw_rect(7, 7, 'grey47')
        self._draw_rect(2, 8, 'grey47')
        self._draw_rect(6, 8, 'grey47')
        # 画出终点方位，并设置为白色
        self._draw_rect(8, 8, 'white')
        # 画出寻路机器人，初始化起始地址，并设置颜色
        self.rect = self._draw_rect(0, 0, 'orangered')
        self.canvas.pack()  # 显示整个画出的窗口

    def _draw_rect(self, x, y, color):#画矩形，x,y表示横、竖第几个格子(从左上到右下)
        padding = 6  # 内边距6px
        coor = [self.UNIT * x + padding, self.UNIT * y + padding, self.UNIT * (x + 1) - padding,
                self.UNIT * (y + 1) - padding]#根据内边距以及每一个格子的坐标得到每一个方块的四个顶点设定
        return self.canvas.create_rectangle(*coor, fill=color)#画出矩形并显示其颜色

    def re_location(self):#定义起点以及是否到终点的done值，便于到达陷阱时机器人归位
        self.x = 0
        self.y = 0
        self.done = False

    def boundary_limit(self, x, y):#纠正坐标值，防止越出边界
        x = max(x, 0)
        x = min(x, 8)
        y = max(y, 0)
        y = min(y, 8)
        return x, y

    def move_to(self,state, delay=0.6):#根据传入的状态,玩家移动到新位置，每0.6更新位置移动
        coor_old = self.canvas.coords(self.rect)  # 得到机器人修改线的位置，便于更新位置
        x, y = state % 9,state // 9  #横竖第几个格子
        padding = 6  # 内边距6px
        coor_new = [self.UNIT * x + padding, self.UNIT * y + padding, self.UNIT * (x + 1) - padding,self.UNIT * (y + 1) - padding]#获取新的修改线
        dx_pixels, dy_pixels = coor_new[0] - coor_old[0], coor_new[1] - coor_old[1]  # 左上角顶点坐标之差
        self.canvas.move(self.rect, dx_pixels, dy_pixels)#个体位置更新，移动
        self.update() #更新
        time.sleep(delay) #机器人每次移动的速度控制

    def Enviroment_interaction(self,action):#动作更新，并得到奖励值
        self.done = False#终点判断
        if action==0:#向上移动
            self.x,self.y = self.x,self.y-1
        elif action==1:#向下移动
            self.x,self.y = self.x,self.y+1
        elif action==2:#向左移动
            self.x,self.y = self.x-1,self.y
        elif action==3:#向右移动
            self.x,self.y = self.x+1,self.y
        self.x, self.y = self.boundary_limit(self.x, self.y)#得到可执行的下一个状态坐标
        #设置陷阱，当到达陷阱时，重置位置(0,0),奖励-100
        if self.x == 0 and self.y == 3:  # 掉入陷阱
            reward = -100
            self.re_location()
        elif self.x == 1 and self.y == 1 or self.x == 1 and self.y == 3 or self.x == 1 and self.y == 5:# 掉入陷阱
            reward = -100
            self.re_location()
        elif self.x == 2 and self.y == 6 or self.x == 2 and self.y == 8 :# 掉入陷阱
            reward = -100
            self.re_location()
        elif self.x == 3 and self.y == 0 or self.x == 3 and self.y == 2 or self.x == 3 and self.y == 3\
                or self.x == 3 and self.y == 5:# 掉入陷阱
            reward = -100
            self.re_location()
        elif self.x == 4 and self.y == 3 or self.x == 4 and self.y == 6 :# 掉入陷阱
            reward = -100
            self.re_location()
        elif self.x == 5 and self.y == 1 or self.x == 5 and self.y == 3 or self.x == 5 and self.y == 7:# 掉入陷阱
            reward = -100
            self.re_location()
        elif self.x == 6 and self.y == 0  or self.x == 6 and self.y == 5 or self.x == 6 and self.y == 8 :# 掉入陷阱
            reward = -100
            self.re_location()
        elif self.x == 7 and self.y == 3 or self.x == 7 and self.y == 5 or self.x == 7 and self.y == 7 :# 掉入陷阱
            reward = -100
            self.re_location()
        elif self.x == 8 and self.y == 2 or self.x == 8 and self.y == 4 :# 掉入陷阱
            reward = -100
            self.re_location()
        elif self.x == 8 and self.y == 8:  # 到达目的地奖励100，done值设为1
            reward = 100
            self.done = True
        else:  #到达非陷阱处，奖励-1
            reward = -1
        return tuple((self.x, self.y)), reward, self.done #返回状态坐标以及奖励值和是否到达终点的done值

class robot():
    def epsilon_greedy_policy(state,Q, epsilon=0.1):  # 定义ε贪婪策略下，最优动作概率函数
        action = np.argmax(Q.loc[state]) #选取Q表该状态下动作价值最大的索引——对应相应的行为
        probability = np.ones(Maze.actions, dtype=np.float64) * epsilon / Maze.actions  #定义四种相同的动作选择概率[0.25,0.25,0.25,0.25]
        probability[action] += 1 - epsilon  #该行为选择概率为1.25-ε，使得其他三种行为被选择的概率相同为0.25
        return probability  # 返回四种动作概率

    def greedy_policy(state,Q):  # 定义贪婪策略下最优动作函数
        best_action = np.argmax(Q.loc[state]) #选取Q表该状态下动作价值最大的索引——对应相应的行为
        return best_action  #返回动作

    def Qlearning(episode=100, alpha=0.2,discount_rate=0.8):  # 定义返回动作价值和奖励的QLearning函数
        env = Maze()  #使用env调用Maze地图类函数
        rewards = []  #定义用于储存奖励值的列表
        Q = pd.DataFrame(np.zeros((Maze.states, Maze.actions)), columns=[0, 1, 2, 3])#初始化创建Q表,(81个状态(行索引),4个动作(列索引)且列索引值对应相应的动作(上下左右))
        for i in range(episode):  #学习次数
            env.re_location()  #定义初始位置
            current_state, done = np.array((0,0)), False  #得到当前位置的坐标和当前位置的done值
            # env.move_to((current_state[0]+current_state[1]*9))  #更新并显示玩家的位置(将坐标转换为状态值(0~80))
            sum_reward = 0.0  # 定义奖励总值并初始化为0
            while not done:  # 如果不是终点
                probability = robot.epsilon_greedy_policy((current_state[0]+current_state[1]*9),Q)#由当前状态得到最有动作概率
                action = np.random.choice(np.arange(4), p=probability)  # 以概率动作probability随机选取一个动作
                next_state, reward, done = env.Enviroment_interaction(action)  #与环境交互得到下一步的动作、奖励与是否到达终点的done值
                if done :
                    Q.loc[(current_state[0]+current_state[1]*9),action] = Q.loc[(current_state[0]+current_state[1]*9),action] + alpha * (
                     reward + discount_rate * 0.0 - Q.loc[(current_state[0]+current_state[1]*9),action])  #根据贝尔曼方程,对Q表进行更新
                    break
                else:
                    next_action = robot.greedy_policy((next_state[0]+next_state[1]*9),Q)  #根据Q表得到该状态下动作价值最大的动作
                    Q.loc[(current_state[0]+current_state[1]*9),action] = Q.loc[(current_state[0]+current_state[1]*9),action] + alpha * (
                     reward + discount_rate * Q.loc[next_state[0]+next_state[1]*9,next_action] - Q.loc[(current_state[0]+current_state[1]*9),action])  #根据贝尔曼方程,对动作价值Q进行更新
                    current_state = next_state#状态更新
                    # env.move_to((current_state[0]+current_state[1]*9)) #更新并显示玩家的位置(将坐标转换为状态值(0~80))
                sum_reward += reward  #储存每一步的奖励值
            rewards.append(sum_reward)  #累加得到一条完整路径的奖励值
            print('Episode {}:奖励={}'.format(i + 1, sum_reward))  # 显示每次学习的奖励值
        torch.save(Q,"q_table666777.pkl")#保存Q表
        return Q, rewards  # 返回Q表和奖励值

    def test_Qlearning(Q):  # 定义返回奖励的test_Qlearning函数
        env = Maze()  #使用env调用Maze地图类函数
        rewards = []  #定义用于储存奖励值的列表
        env.re_location()  #定义初始位置
        current_state, done = np.array((0,0)), False  #得到当前位置的坐标和当前位置的done值
        env.move_to(current_state[0]+current_state[1]*9)  #更新并显示玩家的位置(将坐标转换为状态值(0~80))
        sum_reward = 0.0  # 定义奖励总值并初始化为0
        while not done:  # 如果不是终点
            action = robot.greedy_policy((current_state[0]+current_state[1]*9),Q)#根据Q表得到该状态下动作价值最大的动作
            next_state, reward, done = env.Enviroment_interaction(action)#与环境交互得到下一步的动作、奖励与是否到达终点的done值
            env.move_to(next_state[0] + next_state[1] * 9)  #更新并显示玩家的位置(将坐标转换为状态值(0~80))
            current_state = next_state  #动作更新
            sum_reward += reward  #储存每一步的奖励值
            rewards.append(sum_reward)  #累加得到一条完整路径的奖励值
        print('奖励={}'.format(sum(rewards)))  # 显示测试完成的奖励值
        print(Q)#输出Q表
        return rewards  #返回奖励值

if __name__ == '__main__':
    # _,rewards=robot.Qlearning(episode=500)  #学习,得到总的奖励值列表
    # plt.plot(rewards)  # 画出奖励值图像
    # plt.xlabel("episode")
    # plt.ylabel("reward")
    # plt.show()

    Q=torch.load("q_table666777.pkl")#导入Q表，开始测试
    robot.test_Qlearning(Q)

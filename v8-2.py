# 将执行greedy和随机的条件互换

#!/usr/bin/env python
# coding: utf-8

# # 实验说明
#
# ## 作业说明
#
# ### 目标：
#
# 训练一个玩2048的神经网络，并得到较高的准确率。
#
# ### 背景：
#
# 2048是一个益智小游戏，规则为：控制所有方块向同一个方向运动，两个相同数字方块撞在一起后合并，成为他们的和。每次操作时会随机生成一个2或者4，最终得到一个“2048”的方块就算胜利。规则的直观解释：[Click to Play 2048](https://play2048.co/)
#
# 本教程展示如何训练一个玩2048的神经网络模型，并评估其最终能够得到的分数。

# #### 建模过程：
#
# 2048游戏可以理解为一个这样的过程：
#
# <blockquote>
#
# 有一个**局面（state）**，4x4格子上的一些数字。
#
# <img src="https://data.megengine.org.cn/megstudio/images/2048_demo.png" width=256 height=256 />
#
# 你可以选择做一些**动作（action）**，比如按键盘上的上/下/左/右键。
#
# 你有一些**策略（policy）**，比如你觉得现在按左最好，因为这样有两个8可以合并。对于每个动作，可以用一个打分函数来决定你的策略。
#
# 在按照策略做完动作之后，你会得到一个**奖励（reward）**，比如因为两个8合并，分数增加了16，这个16可以被看作是这一步的奖励。
#
# 在许多步之后，游戏结束，你会得到一个**回报（return）**，即游戏的最终分数。
#
# </blockquote>
#
# 由此，我们将2048建模为一个马尔可夫决策过程，其求解可以通过各种强化学习方法来完成。在baseline中，我们使用了 [Double DQN](https://arxiv.org/abs/1509.06461)。
#
# ### 任务：
#
# Q1：训练模型
#
# 运行baseline，训练和评估模型。观察游戏结束时的滑动平均分数。你可以调用`print_grid`函数输出模型玩游戏的过程，以判断模型是否可以得到合理的结果。
# 提供参考数据：纯随机游玩，平均分数约为570分。在baseline的训练过程中，模型最高可以达到8000分，平均为2000分左右。
#
# 请你修改参数，模型结构等，使得游戏的平均分数尽可能地高。请注意：这里的平均分数指每个游戏结束**最终分数**的平均值。
# **请于q1.diff提交你的代码。**
#
# ## 数据集
#
# 2048游戏代码来源：[console2048](https://github.com/Mekire/console-2048/blob/master/console2048.py)
#
# ## 文件存储
# 实验中生成的文件可以存储于 workspace 目录中。 查看工作区文件，该目录下的变更将会持久保存。 您的所有项目将共享一个存储空间，请及时清理不必要的文件，避免加载过慢。
#
# ## 实验步骤
#
# 1.导入库

# In[1]:


import megengine as mge
import math
import numpy
import numpy as np
import megengine.module as M
import megengine.functional as F
import megengine.data.transform as T
from random import random, randint, shuffle, sample

import simplejson
from megengine.optimizer import Adam
from megengine.autodiff import GradManager
from megengine import tensor
from tqdm import tqdm
import time
import os
import torch
import pickle as pickle
from megengine.core._imperative_rt.utils import Logger
Logger.set_log_level(Logger.LogLevel.Error)

log = open('log_{}.txt'.format(__file__), 'a')
# log = open("log_v8.txt", 'a')
# 2.2048游戏函数

# In[2]:


# https://github.com/Mekire/console-2048/blob/master/console2048.py

def push_left(grid):
    moved, score = False, 0
    rows, columns = grid.shape[0], grid.shape[1]
    for k in range(rows):
        i, last = 0, 0
        for j in range(columns):
            e = grid[k, j]
            if e:
                if e == last:
                    grid[k, i-1]+=e
                    score += e
                    last, moved = 0, True
                else:
                    moved |= (i != j)
                    last=grid[k, i]=e
                    i+=1
        while i<columns:
            grid[k,i]=0
            i+=1
    return score if moved else -1

def push_right(grid):
    moved, score = False, 0
    rows, columns = grid.shape[0], grid.shape[1]
    for k in range(rows):
        i = columns-1
        last  = 0
        for j in range(columns-1,-1,-1):
            e = grid[k, j]
            if e:
                if e == last:
                    grid[k, i+1]+=e
                    score += e
                    last, moved = 0, True
                else:
                    moved |= (i != j)
                    last=grid[k, i]=e
                    i-=1
        while 0<=i:
            grid[k, i]=0
            i-=1
    return score if moved else -1

def push_up(grid):
    moved,score = False, 0
    rows, columns = grid.shape[0], grid.shape[1]
    for k in range(columns):
        i, last = 0, 0
        for j in range(rows):
            e = grid[j, k]
            if e:
                if e == last:
                    score += e
                    grid[i-1, k]+=e
                    last, moved = 0, True
                else:
                    moved |= (i != j)
                    last=grid[i, k]=e
                    i+=1
        while i<rows:
            grid[i, k]=0
            i+=1
    return score if moved else -1

def push_down(grid):
    moved, score = False, 0
    rows, columns = grid.shape[0], grid.shape[1]
    for k in range(columns):
        i, last = rows-1, 0
        for j in range(rows-1,-1,-1):
            e = grid[j, k]
            if e:
                if e == last:
                    score += e
                    grid[i+1, k]+=e
                    last, moved = 0, True
                else:
                    moved |= (i != j)
                    last=grid[i, k]=e
                    i-=1
        while 0<=i:
            grid[i, k]=0
            i-=1
    return score if moved else -1

def push(grid, direction):
    if direction&1:
        if direction&2:
            score = push_down(grid)
        else:
            score = push_up(grid)
    else:
        if direction&2:
            score = push_right(grid)
        else:
            score = push_left(grid)
    return score


def put_new_cell(grid):
    n = 0
    r = 0
    i_s=[0]*16
    j_s=[0]*16
    for i in range(grid.shape[0]):
        for j in range(grid.shape[1]):
            if not grid[i,j]:
                i_s[n]=i
                j_s[n]=j
                n+=1
    if n > 0:
        r = randint(0, n-1)
        grid[i_s[r], j_s[r]] = 2 if random() < 0.9 else 4
    return n

def any_possible_moves(grid):
    """Return True if there are any legal moves, and False otherwise."""
    rows = grid.shape[0]
    columns = grid.shape[1]
    for i in range(rows):
        for j in range(columns):
            e = grid[i, j]
            if not e:
                return True
            if j and e == grid[i, j-1]:
                return True
            if i and e == grid[i-1, j]:
                return True
    return False



def prepare_next_turn(grid):
    """
    Spawn a new number on the grid; then return the result of
    any_possible_moves after this change has been made.
    """
    empties = put_new_cell(grid)
    return empties>1 or any_possible_moves(grid)


def print_grid(grid_array):
    """Print a pretty grid to the screen."""
    print("")
    wall = "+------"*grid_array.shape[1]+"+"
    print(wall)
    for i in range(grid_array.shape[0]):
        meat = "|".join("{:^6}".format(grid_array[i,j]) for j in range(grid_array.shape[1]))
        print("|{}|".format(meat))
        print(wall)


class Game:
    def __init__(self, cols=4, rows=4):
        self.grid_array = np.zeros(shape=(rows, cols), dtype='uint16')
        self.grid = self.grid_array
        for i in range(2):
            put_new_cell(self.grid)
        self.score = 0
        self.end = False

    def copy(self):
        rtn = Game(self.grid.shape[0], self.grid.shape[1])
        for i in range(self.grid.shape[0]):
            for j in range(self.grid.shape[1]):
                rtn.grid[i,j]=self.grid[i,j]
        rtn.score = self.score
        rtn.end = self.end
        return rtn

    def max(self):
        m = 0
        for i in range(self.grid.shape[0]):
            for j in range(self.grid.shape[1]):
                if self.grid[i,j]>m:
                    m = self.grid[i,j]
        return m


    def move(self, direction):
        score = push(self.grid, direction)
        if score == -1:
            return 0
        self.score += score
        if not prepare_next_turn(self.grid):
            self.end = True
        return 1

    def display(self):
        print_grid(self.grid_array)



'''没有使用的函数'''
def random_play(game):
    moves = [0,1,2,3]
    while not game.end:
        shuffle(moves)
        for m in moves:
            if game.move(m):
                break
    return game.score


# 3.定义记忆回放类并实例化
#
# 在记录一次决策过程后，我们存储到该类中，并在训练时选择一部分记忆进行训练。

# In[3]:


# https://github.com/megvii-research/ICCV2019-LearningToPaint/blob/master/baseline/DRL/rpm.py

class SumTree(object):
    # 建立tree和data。
    # 因为SumTree有特殊的数据结构
    # 所以两者都能用一个一维np.array来存储
    data_pointer = 0

    def __init__(self, capacity):
        self.capacity = capacity  # for all priority values
        self.tree = np.zeros(2 * capacity - 1)
        # [--------------Parent nodes-------------][-------leaves to recode priority-------]
        #             size: capacity - 1                       size: capacity
        self.data = np.zeros(capacity, dtype=object)  # for all transitions
        # [--------------data frame-------------]
        #             size: capacity

    #当有新的sample时，添加进tree和data
    def add(self, p, data):
        tree_idx = self.data_pointer + self.capacity - 1
        self.data[self.data_pointer] = data  # update data_frame
        self.update(tree_idx, p)  # update tree_frame

        self.data_pointer += 1
        if self.data_pointer >= self.capacity:  # replace when exceed the capacity
            self.data_pointer = 0

    #当sample被train,有了新的TD-error,就在tree中更新
    def update(self, tree_idx, p):
        change = p - self.tree[tree_idx]
        self.tree[tree_idx] = p
        # then propagate the change through tree
        while tree_idx != 0:    # this method is faster than the recursive loop in the reference code
            tree_idx = (tree_idx - 1) // 2
            self.tree[tree_idx] += change

    #根据选取的v点抽取样本
    def get_leaf(self, v):
        """
        Tree structure and array storage:
        Tree index:
             0         -> storing priority sum
            / \
          1     2
         / \   / \
        3   4 5   6    -> storing priority for transitions
        Array type for storing:
        [0,1,2,3,4,5,6]
        """
        parent_idx = 0
        while True:     # the while loop is faster than the method in the reference code
            cl_idx = 2 * parent_idx + 1         # this leaf's left and right kids
            cr_idx = cl_idx + 1
            if cl_idx >= len(self.tree):        # reach bottom, end search
                leaf_idx = parent_idx
                break
            else:       # downward search, always search for a higher priority node
                if v <= self.tree[cl_idx]:
                    parent_idx = cl_idx
                else:
                    v -= self.tree[cl_idx]
                    parent_idx = cr_idx

        data_idx = leaf_idx - self.capacity + 1
        return leaf_idx, self.tree[leaf_idx], self.data[data_idx]

    @property
    def total_p(self):
        return self.tree[0]  # the root


class Memory(object):  # stored as ( s, a, r, s_ ) in SumTree
    """
    This Memory class is modified based on the original code from:
    https://github.com/jaara/AI-blog/blob/master/Seaquest-DDQN-PER.py
    """
    epsilon = 0.01  # small amount to avoid zero priority
    alpha = 0.6  # [0~1] convert the importance of TD error to priority
    beta = 0.4  # importance-sampling, from initial value increasing to 1
    beta_increment_per_sampling = 0.001
    abs_err_upper = 1.  # clipped abs error

    #建立SumTree和各种参数
    def __init__(self, capacity):
        self.tree = SumTree(capacity)

    #存储数据，更新SumTree
    def store(self, transition):
        max_p = np.max(self.tree.tree[-self.tree.capacity:])
        if max_p == 0:
            max_p = self.abs_err_upper
        self.tree.add(max_p, transition)   # set the max p for new p

    #抽取sample
    def sample(self, n):
        b_idx, b_memory, ISWeights = np.empty((n,), dtype=np.int32), np.empty((n, self.tree.data[0].size)), np.empty((n, 1))
        pri_seg = self.tree.total_p / n       # priority segment
        self.beta = np.min([1., self.beta + self.beta_increment_per_sampling])  # max = 1

        min_prob = np.min(self.tree.tree[-self.tree.capacity:]) / self.tree.total_p     # for later calculate ISweight
        for i in range(n):
            a, b = pri_seg * i, pri_seg * (i + 1)
            v = np.random.uniform(a, b)
            idx, p, data = self.tree.get_leaf(v)
            prob = p / self.tree.total_p
            ISWeights[i, 0] = np.power(prob/min_prob, -self.beta)
            b_idx[i], b_memory[i, :] = idx, data
        return b_idx, b_memory, ISWeights

    #train完被抽取的samples后更新在tree中的sample的priority
    def batch_update(self, tree_idx, abs_errors):
        abs_errors += self.epsilon  # convert to abs and avoid 0
        clipped_errors = np.minimum(abs_errors, self.abs_err_upper)
        ps = np.power(clipped_errors, self.alpha)
        for ti, p in zip(tree_idx, ps):
            self.tree.update(ti, p)

data = Memory(5000)


# 4.定义模型结构

# In[4]:


class Net(M.Module):
    def __init__(self):
        super(Net, self).__init__()
        self.conv1 = M.Conv2d(16, 128, (1,2))
        self.relu1 = M.ReLU()
        self.conv2 = M.Conv2d(16, 128, (2,1))
        self.relu2 = M.ReLU()
        self.conv11 = M.Conv2d(128, 128, (2,1))
        self.relu11 = M.ReLU()
        self.conv12 = M.Conv2d(128, 128, (1,2))
        self.relu12 = M.ReLU()
        self.conv21 = M.Conv2d(128, 128, (2,1))
        self.relu21 = M.ReLU()
        self.conv22 = M.Conv2d(128, 128, (1,2))
        self.relu22 = M.ReLU()
        self.fc1 = M.Linear(7424, 256)
        self.relu_fc1 = M.ReLU()
        self.fc2 = M.Linear(256, 4)
        self.relu_fc2 = M.ReLU()

    def forward(self, x):
        c1 = self.relu1(self.conv1(x))
        c2 = self.relu2(self.conv2(x))
        c11 = self.relu11(self.conv11(c1))
        c12 = self.relu12(self.conv12(c1))
        c21 = self.relu21(self.conv21(c2))
        c22 = self.relu22(self.conv22(c2))
        def batch_flatten(mm):
            return list(map(lambda l: F.flatten(l, 1), mm))
        c = F.concat(batch_flatten([c1, c2, c11, c12, c21, c22]), 1)
        x1 = self.relu_fc1(self.fc1(c))
        x2 = self.relu_fc2(self.fc2(x1))
        x = F.reshape(x2,(-1,4))
        return x

model = Net()
model_target = Net()


# 5.定义输入转化函数，使得局面可以被输入进模型。

# In[5]:


table = {2**i:i for i in range(1,16)}
table[0] = 0

def make_input(grid):
    g0 = grid
    r = np.zeros(shape=(16, 4, 4), dtype=np.float32)
    for i in range(4):
        for j in range(4):
            v = g0[i, j]
            r[table[v], i, j] = 1
    return r

def find_empty(grid):
    g0 = grid
    count = 0
    for i in range(4):
        for j in range(4):
            if g0[i, j] == 0:
                count += 1
    return count

# 6.定义优化器

# In[6]:
def sard_to_transition(s0, s1, a, reward, done):
    s0_1 = np.reshape(s0, s0.size)
    s1_1 = np.reshape(s1, s1.size)
    a_1 = np.reshape(a, a.size)
    reward_tmp = np.array(reward)
    reward_1 = np.reshape(reward_tmp, reward_tmp.size)
    done_tmp = np.array(done, dtype=int)
    done_1 = np.reshape(done_tmp, done_tmp.size)
    return np.hstack((s0_1, s1_1, a_1, reward_1, done_1))

def transition_to_sard(transition):
    # s0, s1, a, reward, done
    # 16*4*4, 16*484, 1, 1, 1
    # 256, 256, 1, 1, 1
    s0_1 = transition[:, :256]
    s0_2 = np.reshape(s0_1, [games_count, 16, 4, 4]).astype(int)
    s1_1 = transition[:, 256: 512]
    s1_2 = np.reshape(s1_1, [games_count, 16, 4, 4]).astype(int)
    a_1 = transition[:, 512]
    a_2 = np.reshape(a_1, [games_count, 1]).astype(int)
    reward_1 = transition[:, 513]
    reward_2 = np.reshape(reward_1, [games_count, 1]).astype(int)
    done_1 = transition[:, 514]
    done_2 = np.reshape(done_1, [games_count, 1]).astype(bool)
    return tensor(s0_2), tensor(s1_2), tensor(a_2), tensor(reward_2), tensor(done_2)


opt = Adam(model.parameters(), lr=1e-4, clipnorm=1.)


# 7.模型训练

# In[7]:


maxscore = 0
avg_score = 0
epochs = 20000
games_count = 32

game = []
'''Play 32 games at the same time'''
for i in range(games_count):
    game.append(Game())

with tqdm(total=epochs*5, desc="epoch") as tq:
    for epoch in range(epochs):

        '''double DQN'''
        if epoch % 10 == 0:
            mge.save(model, "1.mge")
            model_target = mge.load("1.mge")

        grid = []
        for k in range(games_count):

            '''Check if the game is over'''
            if any_possible_moves(game[k].grid) == False:
                if avg_score == 0:
                    avg_score = game[k].score
                else:
                    avg_score = avg_score * 0.99 + game[k].score * 0.01
                game[k] = Game()

            tmp = make_input(game[k].grid)
            grid.append(tensor(tmp))

        status = F.stack(grid, 0)

        '''Choose the action with the highest probability'''
        values = model(status).detach()
        choice_greedy = F.argmax(values, 1)
        a = numpy.zeros([values.shape[0]], dtype=int)
        e = 1. / (epoch / 100 + 1)
        if epoch >= epochs * 0.75:
            e = 0
        for k in range(0, games_count):
            random_num = random()
            if random_num >= e:
                a[k] = int(choice_greedy[k])
            else:
                a[k] = int(randint(0, values.shape[1] - 1))

        for k in range(games_count):
            pre_score = game[k].score
            pre_grid = game[k].grid.copy()
            game[k].move(a[k])
            after_score = game[k].score
            if game[k].score > maxscore:
                maxscore = game[k].score
            action = a[k]

            '''In some situations, some actions are meaningless, try another'''
            while (game[k].grid == pre_grid).all():
                action = (action + 1) % 4
                game[k].move(action)

            score = after_score - pre_score
            done = tensor(any_possible_moves(game[k].grid) == False)
            z = np.count_nonzero(game[k].grid) - np.count_nonzero(pre_grid)
            z = -z
            z += 1
            m_after = np.max(game[k].grid)
            m_before = np.max(pre_grid)
            if m_after > m_before:
                z += math.log(m_after, 2) * 0.1
            empty_after = find_empty(game[k].grid)
            empty_before = find_empty(pre_grid)
            if empty_after > empty_before and empty_after != 0:
                z += math.log(empty_after, 2) * 0.5

            # print("Merged: ", z, m_after)
            grid = tensor(make_input(game[k].grid.copy()))

            '''Record to memory'''
            '''(status, next_status, action, score, if_game_over)'''
            # data.store((tensor(make_input(pre_grid)), tensor(grid), tensor(a[k]), tensor(z), done))
            tmp = sard_to_transition(make_input(pre_grid), grid, a[k], z, done)
            data.store((tensor(tmp)))

        for i in range(5):
            gm = GradManager().attach(model.parameters())
            with gm:
                idx, memory, w = data.sample(games_count)
                s0, s1, a, reward, d = transition_to_sard(memory)
                '''double DQN'''
                pred_s0 = model(s0)
                pred_s1 = F.max(model_target(s1), axis=1)

                loss = 0
                total_Q = 0
                total_reward = 0
                abs_errors = np.zeros([games_count])
                for i in range(games_count):
                    Q = pred_s0[i][a[i]]
                    total_Q += Q
                    total_reward += reward[i]
                    tmp = F.loss.square_loss(Q, pred_s1[i].detach() * 0.99 * (1 - d[i]) + reward[i])
                    loss += tmp
                    abs_errors[i] = tmp

                data.batch_update(idx, abs_errors)

                loss /= games_count
                total_Q /= games_count
                total_reward = total_reward / games_count * 128
                tq.set_postfix(
                    {
                        "loss": "{0:1.5f}".format(loss.numpy().item()),
                        "Q": "{0:1.5f}".format(total_Q.numpy().item()),
                        "reward": "{0:1.5f}".format(total_reward.numpy().item()),
                        "avg_score":"{0:1.5f}".format(avg_score),
                    }
                )
                tq.update(1)
                gm.backward(loss)

            opt.step()
            opt.clear_grad()
        if epoch % 100 == 0:
            my_dic = {
                "loss": "{0:1.5f}".format(loss.numpy().item()),
                "Q": "{0:1.5f}".format(total_Q.numpy().item()),
                "reward": "{0:1.5f}".format(total_reward.numpy().item()),
                "avg_score":"{0:1.5f}".format(avg_score),
            }
            log.write(str(epoch * 5) + ":" + simplejson.dumps(my_dic) + "\n")
            log.flush()
log.close()
print("maxscore:{}".format(maxscore))
print("avg_score:{}".format(avg_score))




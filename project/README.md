## Deep Reinforcement Learning

### 简介

这是2023春季学期，SJTU CS3316强化学习课程的课程设计



### 目录结构说明



+ value-based/：value-based算法目录

  + main.py：main文件，读取参数，实例化一个agent，开启训练
  + agents.py：按不同的算法，分别实现了不同的agent，agent通过`step()`和`act()`与环境进行交互

  + model.py：集成了不同算法的agent，提供一个统一的`train()`接口
  + network.py：实现了算法所需的网络结构
  + utils.py：实现了一些辅助类和函数，如`OUNoise`，`RunningMeanStd`，`get_device`和`ReplayBuffer`等

+ policy-based/：policy-based算法目录

  + main.py：main文件，读取参数，实例化一个agent，开启训练
  + agents.py：按不同的算法，分别实现了不同的agent，agent通过`step()`和`act()`与环境进行交互

  + model.py：集成了不同算法的agent，提供一个统一的`train()`接口
  + network.py：实现了算法所需的网络结构
  + utils.py：实现了一些辅助类和函数，如`get_device`，`ReplayBuffer`和`PrioritizedReplayBuffer`等

+ run.py：脚本文件，可直接按默认参数设置，选择不同的环境和算法，运行代码

+ atari.yaml：Atari环境的conda环境配置文件

+ mujoco.yaml：Mujoco环境的conda环境配置文件

+ README.md：README文档



### 测试方式

在测试前，请确保所需依赖项都已安装完毕，具体可见环境配置文件`atari.yaml`和`mujoco.yaml`，也可以在已安装anaconda的情况下，使用如下命令创建一个新的可用环境：

~~~shell
conda env create -f ./atari.yaml
~~~

~~~shell
conda env create -f ./mujoco.yaml
~~~

注意这两个系列的gym仿真环境，最好使用不同的虚拟环境来配置，以避免冲突



依赖项安装完毕后，可进入`code/`目录，运行以下命令：

~~~shell
python ./run.py --env_name <env_name> -m <model_type>
~~~

`run.py `文件接收三个参数：

+ `--env_name`：待测试环境名，默认为`'VideoPinball-ramNoFrameskip-v4'`

+ `'-m','--model_type'`：待测试的算法名称，默认为`DQN`
+ `'-c','--cuda'`：使用的cuda编号，-1代表使用CPU
+ `--per`：针对Value-based算法有效，若添加此参数，则使用Priority Replaybuffer，默认为False
+ `--noisy`：针对Value-based算法有效，若添加此参数，则使用Noisy Layer代替epsilon greedy，默认为False



运行后，程序输出结果会以“env_name+model_type.txt”的形式保存
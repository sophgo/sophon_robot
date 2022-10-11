本仓库包含了一些适用算能SE5微服务器（或其他使用BM1684芯片开发板）的ROS代码示例，  
目前代码库支持基于摄像头的相关demo包含，   
0.颜色检测、边缘检测等 1.人体检测 2.人物跟踪 3.人脸检测和人脸识别 4.人体关键点检测 5.动作检测 6种demo。

## sophon_robot包的安装和使用

（1）进入到SE5的/data目录下，在/data 下新建一个文件夹 workspace， 将workspace 软连接到 ～目录下

```bash
mkdir /data/workspace
ln -s /data/workspace ~
cd ~
ls
```

（2）构建代码

新建目录~/workspace/robot_ws/src

```bash
cd workspace
mkdir -p robot_ws/src
```

将源代码拷贝到~/workspace/robot_ws/src目录下

```bash
cd ~/workspace/robot_ws/src
git clone https://github.com/sophgo/sophon_robot
```


运行catkin_make命令构建代码

```
cd ~/workspace/robot_ws
catkin_make
```

（3）创建功能包，给代码执行权限

```bash
echo "source ~/workspace/robot_ws/devel/setup.bash" >> ~/.bashrc
source ~/.bashrc
sudo chmod +x /data/workspace/robot_ws/src/sophon_robot/scripts/cv/*
```


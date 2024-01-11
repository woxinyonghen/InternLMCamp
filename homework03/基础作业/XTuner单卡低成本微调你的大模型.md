1 创建环境



```bash
bash

# 创建环境
conda create --name personal_assistant python=3.10 -y

# 激活环境
conda activate personal_assistant

cd /root/
mkdir /root/personal_assistant && cd /root/personal_assistant

# 拉取0.1.9的版本源码
git clone -b v0.1.9  https://github.com/InternLM/xtuner

# 进入源码目录
cd xtuner

# 从源码安装 XTuner
pip install -e '.[all]'
```



2 数据准备

在`data`目录下创建一个json文件`personal_assistant.json`作为本次微调所使用的数据集。

```bash
mkdir /root/personal_assistant/data && cd /root/personal_assistant/data

touch personal_assistant.json
```

personal_assistant.json内容如下：复制多次数据增强

```
[
    {
        "conversation": [
            {
                "input": "请介绍一下你自己",
                "output": "我是星辰的小助手，内在是上海AI实验室书生·浦语的7B大模型哦"
            }
        ]
    },
    {
        "conversation": [
            {
                "input": "请做一下自我介绍",
                "output": "我是星辰的小助手，内在是上海AI实验室书生·浦语的7B大模型哦"
            }
        ]
    }
]
```



3 配置准备



```
# 下载模型InternLM-chat-7B
mkdir -p /root/personal_assistant/model/Shanghai_AI_Laboratory
cp -r /root/share/temp/model_repos/internlm-chat-7b /root/personal_assistant/model/Shanghai_AI_Laboratory

# 创建用于存放配置的文件夹config并进入
mkdir /root/personal_assistant/config && cd /root/personal_assistant/config

# 列出所有内置配置
xtuner list-cfg

# 拷贝一个配置文件到当前目录：xtuner copy-cfg ${CONFIG_NAME} ${SAVE_PATH}
xtuner copy-cfg internlm_chat_7b_qlora_oasst1_e3 .
```

![image-20240110212029155](C:/Users/HeHang/AppData/Roaming/Typora/typora-user-images/image-20240110212029155.png)

修改拷贝后的文件internlm_chat_7b_qlora_oasst1_e3_copy.py，修改下述位置：

```BASH
# PART 1 中
# 预训练模型存放的位置
pretrained_model_name_or_path = '/root/personal_assistant/model/Shanghai_AI_Laboratory/internlm-chat-7b'

# 微调数据存放的位置
data_path = '/root/personal_assistant/data/personal_assistant.json'

# 训练中最大的文本长度
max_length = 512

# 每一批训练样本的大小
batch_size = 2

# 最大训练轮数
max_epochs = 3

# 验证的频率
evaluation_freq = 90

# 用于评估输出内容的问题（用于评估的问题尽量与数据集的question保持一致）
evaluation_inputs = [ '请介绍一下你自己', '请做一下自我介绍' ]


# PART 3 中
dataset=dict(type=load_dataset, path='json', data_files=dict(train=data_path))
dataset_map_fn=None
```

PART1 部分

![image-20240111104416154](C:/Users/HeHang/AppData/Roaming/Typora/typora-user-images/image-20240111104416154.png)

PART3 部分

![image-20240110212656150](C:/Users/HeHang/AppData/Roaming/Typora/typora-user-images/image-20240110212656150.png)

4 微调启动

```
xtuner train /root/personal_assistant/config/internlm_chat_7b_qlora_oasst1_e3_copy.py
```

![image-20240110232049441](C:/Users/HeHang/AppData/Roaming/Typora/typora-user-images/image-20240110232049441.png)



5 参数转换

训练后的pth格式参数转Hugging Face格式

```bash
# 创建用于存放Hugging Face格式参数的hf文件夹
mkdir /root/personal_assistant/config/work_dirs/hf

export MKL_SERVICE_FORCE_INTEL=1

# 配置文件存放的位置
export CONFIG_NAME_OR_PATH=/root/personal_assistant/config/internlm_chat_7b_qlora_oasst1_e3_copy.py

# 模型训练后得到的pth格式参数存放的位置
export PTH=/root/personal_assistant/config/work_dirs/internlm_chat_7b_qlora_oasst1_e3_copy/epoch_3.pth

# pth文件转换为Hugging Face格式后参数存放的位置
export SAVE_PATH=/root/personal_assistant/config/work_dirs/hf

# 执行参数转换
xtuner convert pth_to_hf $CONFIG_NAME_OR_PATH $PTH $SAVE_PATH
```

![image-20240110213757178](C:/Users/HeHang/AppData/Roaming/Typora/typora-user-images/image-20240110213757178.png)



6 参数合并



```bash
export MKL_SERVICE_FORCE_INTEL=1
export MKL_THREADING_LAYER='GNU'

# 原始模型参数存放的位置
export NAME_OR_PATH_TO_LLM=/root/personal_assistant/model/Shanghai_AI_Laboratory/internlm-chat-7b

# Hugging Face格式参数存放的位置
export NAME_OR_PATH_TO_ADAPTER=/root/personal_assistant/config/work_dirs/hf

# 最终Merge后的参数存放的位置
mkdir /root/personal_assistant/config/work_dirs/hf_merge
export SAVE_PATH=/root/personal_assistant/config/work_dirs/hf_merge

# 执行参数Merge
xtuner convert merge \
    $NAME_OR_PATH_TO_LLM \
    $NAME_OR_PATH_TO_ADAPTER \
    $SAVE_PATH \
    --max-shard-size 2GB
```

![image-20240110233324329](C:/Users/HeHang/AppData/Roaming/Typora/typora-user-images/image-20240110233324329.png)

7 Web部署

```
# 安装依赖
pip install streamlit==1.24.0

# 创建code文件夹用于存放InternLM项目代码
mkdir /root/personal_assistant/code && cd /root/personal_assistant/code
git clone https://github.com/InternLM/InternLM.git

# 修改/root/code/InternLM/web_demo.py中的模型路径
修改为"/root/personal_assistant/config/work_dirs/hf_merge"

# 运行脚本
streamlit run /root/personal_assistant/code/InternLM/web_demo.py --server.address 127.0.0.1 --server.port 6006

# powershell
ssh -CNg -L 6006:127.0.0.1:6006 root@ssh.intern-ai.org.cn -p [开发机端口号]
```

![image-20240110214755134](C:/Users/HeHang/AppData/Roaming/Typora/typora-user-images/image-20240110214755134.png)

8 最终效果

变成星辰的小助手啦

![image-20240111151050783](C:/Users/HeHang/AppData/Roaming/Typora/typora-user-images/image-20240111151050783.png)

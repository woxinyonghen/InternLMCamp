1 创建环境

```
bash
conda create --name opencompass --clone=/root/share/conda_envs/internlm-base
conda activate opencompass
git clone https://github.com/open-compass/opencompass
cd opencompass
pip install -e .
```

2 数据准备

```
# 解压评测数据集到data/处
cp /share/temp/datasets/OpenCompassData-core-20231110.zip /root/opencompass/
unzip OpenCompassData-core-20231110.zip

# 将会在opencompass下看到data文件夹
```



3 查看支持的数据集和模型

```
python tools/list_configs.py internlm ceval
```

![image-20240119233247685](C:/Users/HeHang/AppData/Roaming/Typora/typora-user-images/image-20240119233247685.png)

4 启动评测

```
python run.py \
	--datasets ceval_gen \
	--hf-path /root/share/model_repos/internlm2-chat-7b/ \
	--tokenizer-path /root/share/model_repos/internlm2-chat-7b/ \
	--tokenizer-kwargs padding_side='left' truncation='left' trust_remote_code=True \
	--model-kwargs trust_remote_code=True device_map='auto' \
	--max-seq-len 2048 \
	--max-out-len 16 \
	--batch-size 4 \
	--num-gpus 1 \
```

5 可视化结果展示

![image-20240120185508149](C:/Users/HeHang/AppData/Roaming/Typora/typora-user-images/image-20240120185508149.png)

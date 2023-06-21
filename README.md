# vad_asr

## 使用说明

### 0. 环境搭建
(如果是windows平台，推荐使用wsl)
先安装并创建一个python环境，运行
```shell
git clone http://git.xmov.ai/dujing/vad-asr.git
cd vad-asr
pip install -r requirements.txt -i https://pypi.tuna.tsinghua.edu.cn/simple
```

### 1. 数据准备
创建data/input文件夹，将原始需要切分的长音频放置到此文件夹下

根据音频内容修改热词列表文件hotwords.txt，可以略微改善热词识别效果

### 2. 切句和解码
运行
```shell
./run_vad_asr.sh
```

### 3. 输出数据
在data/output下可以得到切分后音频路径wav.scp，以及对应的转写结果asr.txt

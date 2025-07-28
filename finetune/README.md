# 微调 Qwen/Qwen2-1.5B

```python
pip install -r requirements.txt 
python download.py  #下载数据集并转alpaca格式
python train.py     #训练模型
python generate.py  #对比base模型和微调后到模型输出（如下图，可以看到模型具有思考能力了）
```

![alt text](image.png)



微调具体的参数设置需要根据显卡修改，本微调脚本（train.py）采用混合精度训练，使用adamw优化器，batchsize=1，seqlen=2048，不开gradient_checkpointing。显存占用50多GB（在单张H20下可正确运行）


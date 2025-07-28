import random
import json
from sklearn.model_selection import train_test_split
from modelscope.msdatasets import MsDataset
import logging
import os
 
logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')



#处理为训练输入格式
def transfer(origin_path, new_path):
    PROMPT = "你是一个医学专家，你需要根据用户的问题，给出带有思考的回答。"
    messages = []

    # 读取旧的JSONL文件
    with open(origin_path, "r") as file:
        for line in file:
            # 解析每一行的json数据
            data = json.loads(line)
            input = data["question"]
            think = data["think"]
            answer = data["answer"]
            output = f"<think>{think}</think> \n {answer}"
            message = {
                "instruction": PROMPT,
                "input": f"{input}",
                "output": output,
            }
            messages.append(message)

    # 保存重构后的JSONL文件
    with open(new_path, "w", encoding="utf-8") as file:
        for message in messages:
            file.write(json.dumps(message, ensure_ascii=False) + "\n")



def download(
    dataset_name: str = 'krisfu/delicate_medical_r1_data',
    subset_name: str = 'default',
    split: str = 'train',
    train_ratio: float = 0.9,
    train_output_path: str = 'train_alpaca.jsonl',
    val_output_path: str = 'val_alpaca.jsonl',
    random_seed: int = 42
):
    try:
        # 设置随机种子
        random.seed(random_seed)

        # 加载数据集
        logging.info(f"正在加载数据集: {dataset_name}")
        ds = MsDataset.load(dataset_name, subset_name=subset_name, split=split)

        # 转换为列表
        data_list = list(ds)
        logging.info(f"数据集加载完成，共 {len(data_list)} 条数据")

        # 打乱数据
        logging.info("正在打乱数据...")
        random.shuffle(data_list)

        # 划分训练集和验证集
        logging.info(f"正在按比例 {train_ratio} 划分训练集和验证集...")
        train_data, val_data = train_test_split(
            data_list,
            test_size=1 - train_ratio,
            random_state=random_seed
        )

        # 保存训练集
        tmp_train_path = "./tmp_train.jsonl"
        with open(tmp_train_path, 'w', encoding='utf-8') as f:
            for item in train_data:
                json.dump(item, f, ensure_ascii=False)
                f.write('\n')

        # 保存验证集
        tmp_val_path = "./tmp_val.jsonl"
        with open(tmp_val_path, 'w', encoding='utf-8') as f:
            for item in val_data:
                json.dump(item, f, ensure_ascii=False)
                f.write('\n')

        # 输出结果信息
        logging.info("数据集划分与保存已完成")
        logging.info(f"训练集大小：{len(train_data)}")
        logging.info(f"验证集大小：{len(val_data)}")

        #转finetune所需格式
        logging.info("将数据集转为训练所需格式")
        transfer(tmp_train_path,train_output_path)
        transfer(tmp_val_path,val_output_path)
        os.remove(tmp_train_path)
        os.remove(tmp_val_path)
        logging.info("转换成功")

        print("\n")
        logging.info(f"训练集位置：{train_output_path}")
        logging.info(f"验证集位置：{val_output_path}")

  
    except Exception as e:
        logging.error(f"执行过程中发生错误：{e}", exc_info=True)
        raise


if __name__ == "__main__":
    download()
    




import torch
from transformers import AutoModelForCausalLM, AutoTokenizer

# 指定模型ID
local_model_path ='E:\model\models\Qwen\Qwen3-0___6B'

# 设置设备，优先使用GPU
device = "cuda" if torch.cuda.is_available() else "cpu"
print(f"Using device: {device}")

# 加载分词器
tokenizer = AutoTokenizer.from_pretrained(local_model_path)

# 加载模型，并将其移动到指定设备
model = AutoModelForCausalLM.from_pretrained(local_model_path, torch_dtype="auto", device_map="auto")

print("模型和分词器加载完成！")

# 准备对话输入
messages = [
    {"role": "system", "content": "You are a helpful story writer."},
    {"role": "user", "content": "你好，请写一个小故事。"}
]

# 使用分词器的模板格式化输入
text = tokenizer.apply_chat_template(
    messages,
    tokenize=False,
    add_generation_prompt=True,
    enable_thinking=False  # True is the default value for enable_thinking
)

# 编码输入文本
model_inputs = tokenizer([text], return_tensors="pt").to(device)
print("编码后的输入文本:")
print(model_inputs)

# 使用模型生成回答
# max_new_tokens 控制了模型最多能生成多少个新的Token

generated_ids = model.generate(
    model_inputs.input_ids,
    max_new_tokens=512,
    temperature=1.9,
    top_p=0.9,
    repetition_penalty=1.2
)

# 将生成的 Token ID 截取掉输入部分
# 这样我们只解码模型新生成的部分
generated_ids = [
    output_ids[len(input_ids):] for input_ids, output_ids in zip(model_inputs.input_ids, generated_ids)
]

# 解码生成的 Token ID
response = tokenizer.batch_decode(generated_ids, skip_special_tokens=True)[0]

print("\n模型的回答:")
print(response)
from transformers import AutoTokenizer, AutoModelForCausalLM
import torch

model_path = "meta-llama/Meta-Llama-3-8B-Instruct"
revision = "refs/pr/4"

# GPU 사용 확인
if torch.cuda.is_available():
    print("CUDA is available. Using GPU.")
else:
    print("CUDA is not available. Using CPU.")

# 토크나이저와 모델 로드
tokenizer = AutoTokenizer.from_pretrained(model_path, revision=revision)
model = AutoModelForCausalLM.from_pretrained(model_path, revision=revision, device_map="auto", torch_dtype=torch.bfloat16)

# GPU 장치 설정
device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")

# 프롬프트 설정
prompt = "데이터베이스에서 모든 열을 검색하는 쿼리문 알려줘"

chat = [{'content': prompt, 'role': 'user'}]
chat_tokens = tokenizer.apply_chat_template(chat, tokenize=True, add_generation_prompt=True, return_tensors='pt').to(device)

# 텍스트 생성
new_chat_tokens = model.generate(chat_tokens, do_sample=False, max_new_tokens=128)
new_chat_str = tokenizer.decode(new_chat_tokens[0])
print(new_chat_str)

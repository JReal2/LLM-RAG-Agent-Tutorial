from transformers import AutoTokenizer, AutoModelForCausalLM

# 사전 학습모델 및 토크나이저 로딩
model = AutoModelForCausalLM.from_pretrained('model-name')
tokenizer = AutoTokenizer.from_pretrained('model-name')

# 토큰 추가
new_tokens = ['newword1', 'newword2']
tokenizer.add_tokens(new_tokens)

# 임베딩 공간 리사이즈 
model.resize_token_embeddings(len(tokenizer))

# 추가된 토큰과 함께 파인튜닝. 
# (fine-tuning code here)
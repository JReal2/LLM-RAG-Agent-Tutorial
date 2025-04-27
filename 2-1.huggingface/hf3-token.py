from transformers import BertTokenizer

tokenizer = BertTokenizer.from_pretrained("bert-base-uncased")

# 사전 정의된 특수 토큰 확인
print("CLS Token:", tokenizer.cls_token)  # [CLS]
print("SEP Token:", tokenizer.sep_token)  # [SEP]
print("MASK Token:", tokenizer.mask_token)  # [MASK]
print("PAD Token:", tokenizer.pad_token)  # [PAD]
print("UNK Token:", tokenizer.unk_token)  # [UNK]


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
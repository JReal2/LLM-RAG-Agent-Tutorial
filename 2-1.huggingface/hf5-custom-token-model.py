from transformers import BertTokenizerFast, BertModel
import torch
from torch import nn

# BERT 토크나이저 사전학습모델 로딩
tokenizer = BertTokenizerFast.from_pretrained('bert-base-uncased')
print(tokenizer.tokenize("[CLS] Hello world, how are you?"))

print(tokenizer.tokenize("[newtoken] Hello world, how are you?"))
tokenizer.add_tokens(['[newtoken]'])

# 토큰을 추가하고 다시 토큰화를 한다.
tokenizer.add_tokens(['[newtoken]'])
tokenizer.tokenize("[newtoken] Hello world, how are you?")

# 토큰값을 확인해 본다.
tokenized = tokenizer("[newtoken] Hello world, how are you?", add_special_tokens=False, return_tensors="pt")
print(tokenized['input_ids'])

tkn = tokenized['input_ids'][0, 0]
print("First token:", tkn)
print("Decoded:", tokenizer.decode(tkn))

model = BertModel.from_pretrained('bert-base-uncased')
print(model.embeddings)

try:
    out = model(**tokenized)
    out.last_hidden_state
except Exception as e:
    print(e)

# 임베딩 모델이 추가된 토큰을 학습하지 않았으므로, out of range 에러가 출력될 것이다. 
# 다음 코드로 BERT 모델의토큰 공간 크기를 확인해 본다.
weights = model.embeddings.word_embeddings.weight.data
print(weights.shape)

# 출력은 다음과 같이 30522이다.
torch.Size([30522, 768])

# 이제 [CLS] 토큰을 임베딩 모델에 추가해보자. 
new_weights = torch.cat((weights, weights[101:102]), 0)
new_emb = nn.Embedding.from_pretrained(new_weights, padding_idx=0, freeze=False)
print(new_emb)

# 다음과 같이 30523으로 토큰 크기가 증가되었다. 
Embedding(30523, 768, padding_idx=0)

# 새 레이어를 모델 마지막에 추가한다.
model.embeddings.word_embeddings = new_emb
print(model.embeddings)

# 그 결과로 임베딩 모델의 word_embeddings가 업데이트된다.
BertEmbeddings(
  (word_embeddings): Embedding(30523, 768, padding_idx=0)
  (position_embeddings): Embedding(512, 768)
  (token_type_embeddings): Embedding(2, 768)
  (LayerNorm): LayerNorm((768,), eps=1e-12, elementwise_affine=True)
  (dropout): Dropout(p=0.1, inplace=False)
)

# 앞의 토큰 시퀀스 리스트를 입력한다. 그럼, 제대로 결과가 출력될 것이다.
out = model(**tokenized)
print(out.last_hidden_state)

# 다음 코드를 실행하면, 추가된 모델이 동일한 결과를 가지는 것을 알 수 있다.
model = BertModel.from_pretrained('bert-base-uncased')
out2 = model(
    **tokenizer("[CLS] Hello world, how are you?", add_special_tokens=False, return_tensors="pt")
)

out3 = torch.all(out.last_hidden_state == out2.last_hidden_state)
print(out3)
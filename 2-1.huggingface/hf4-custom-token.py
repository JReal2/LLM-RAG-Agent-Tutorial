from tokenizers import Tokenizer, models, pre_tokenizers, trainers

tokenizer = Tokenizer(models.BPE()) # 토큰화 얻기

tokenizer.pre_tokenizer = pre_tokenizers.Whitespace() # 사용자 토큰 처리 객체
def custom_pre_tokenizer(sequence): # 사용자 토큰 정의
    # Define rules to combine tokens, e.g., "new word" -> "newword"
    combined_sequence = sequence.replace("new word", "newword")
    return combined_sequence

# 토큰 훈련. custom pre-tokenizer 활용함.
trainer = trainers.BpeTrainer()
tokenizer.train(files=["path/to/training/data.txt"], trainer=trainer, pre_tokenizer=custom_pre_tokenizer)

# 훈련된 토큰 저장
tokenizer.save("path/to/customized_tokenizer.json")

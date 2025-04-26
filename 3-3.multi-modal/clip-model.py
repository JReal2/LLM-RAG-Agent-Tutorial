from torch.utils.data import DataLoader
from datasets import load_dataset
from torchvision import transforms
from PIL import Image
import torch
from torchvision import transforms
from PIL import Image

class Flickr30kDataset(torch.utils.data.Dataset): # flickr30k 데이터셋
    def __init__(self):
        self.dataset = load_dataset("nlphuji/flickr30k", cache_dir="./huggingface_data") # 허깅페이스의 데이터획득
        self.transform = transforms.Compose([
            transforms.Resize((224, 224)),
            transforms.ToTensor(),
        ])  # 학습을 위해 이미지 224,224 해상도로 정규화
        self.cap_per_image = 2  # 인코딩될 텍스트 캡션은 2개만 사용

    def __len__(self):
        return self.dataset.num_rows["test"] * self.cap_per_image  # 학습 데이터수=test 이미지들 갯수 x 해당 이미지의 캡션 수

    def __getitem__(self, idx):
        original_idx = idx // self.cap_per_image  # 입력 데이터 인덱스에 대한 이미지 당 캡션수의 몫 획득
        image = self.dataset["test"][original_idx]["image"].convert("RGB")  
        image = self.transform(image)              # 224x224 텐서로 이미지 변환
        caption = self.dataset["test"][original_idx]["caption"][idx % self.cap_per_image]  # 캡션 획득

        return {"image": image, "caption": caption}  # 캡션에 대한 해당 이미지, 텍스트 리턴

flickr30k_custom_dataset = Flickr30kDataset()


from dataclasses import dataclass

@dataclass
class Config:
    embed_dim: int = 512  # 임베딩 차원 
    transformer_embed_dim: int = 768  # 트랜스포머 임베딩 차원
    max_len: int = 32                        # 텍스트 최대 길이 
    text_model: str = "distilbert-base-multilingual-cased"  # 텍스트 특징 추출 인코더 
    epochs: int = 3                           # 학습 에포크
    batch_size: int = 128                    # 배치 크기 

clip_dataloader = DataLoader(flickr30k_custom_dataset, batch_size=BATCH_SIZE, shuffle=True, num_workers=4)  # 데이터 로더 객체 생성

model = CustomModel().to(device)
optimizer = torch.optim.Adam([
    {'params': model.vision_encoder.parameters()},
    {'params': model.caption_encoder.parameters()}
], lr=model.lr)

batch_zero = True
for epoch in range(start_epoch, num_epochs):  # 에포크 횟수만큼 학습
    model.train()
    for batch in clip_dataloader:                     # 미니 배치 학습
        image = batch["image"].to(device)   
        text = batch["caption"]
        loss, img_acc, cap_acc = model.common_step((image, text))   # 이미지-텍스트 쌍 유사도 계산

        optimizer.zero_grad()   # 역전파 학습
        loss.backward()
        optimizer.step()

        if batch_zero:             # 손실값 출력
            print(f"Epoch [{0}/{num_epochs}], Batch Loss: {loss.item()}")
            batch_zero = False

    print(f"Epoch [{epoch+1}/{num_epochs}], Batch Loss: {loss.item()}")
print("Training complete.")
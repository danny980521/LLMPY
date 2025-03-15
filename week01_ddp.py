# 본 코드는 https://pytorch.org/tutorials/intermediate/ddp_tutorial.html의 elastic_ddp.py를 기반으로 작성되었습니다.
# torchrun --nproc_per_node=2 /data/di-LLM_458/denver.in/LLM/main.py

import os
import time

import torch
import torch.distributed as dist
import torch.nn as nn
import torch.optim as optim
from datasets import load_dataset
from torch.nn.parallel import DistributedDataParallel as DDP
from torch.utils.data import DataLoader
from torch.utils.data.distributed import DistributedSampler
from transformers import (AutoModelForCausalLM, AutoTokenizer,
                          DataCollatorForLanguageModeling)

# 현재 프로세스가 사용할 GPU 디바이스 설정 (Local Rank 이용)
torch.cuda.set_device(int(os.environ["LOCAL_RANK"]))

# 분산 처리 환경 초기화 (엔비디아의 NCCL 백엔드 사용)
dist.init_process_group("nccl")

# 현재 프로세스의 rank 확인 (= Global Rank)
rank = dist.get_rank()
print(f"Start running basic DDP example on rank {rank}.")

# 현재 프로세스가 사용할 GPU의 ID 계산
device_id = rank % torch.cuda.device_count()

# 모델 & 토크나이저 불러오기
model_name = "/data/di-LLM_458/denver.in/LLM/models/Llama-3.2-1B"
model = AutoModelForCausalLM.from_pretrained(
    model_name,
    torch_dtype="auto",
    cache_dir="/data/di-LLM_458/denver.in/LLM/.cache",
)
tokenizer = AutoTokenizer.from_pretrained(model_name)
tokenizer.pad_token = tokenizer.eos_token

# DDP에 맞추어 모델 이동 & 래핑
# torch.nn.parallel.DistributedDataParallel: 각 model replica에 대해 gradient를 동기화하여 data parallelism을 제공하는 모듈
# 참고: https://pytorch.org/docs/stable/generated/torch.nn.parallel.DistributedDataParallel.html#torch.nn.parallel.DistributedDataParallel
model.to(device_id)
ddp_model = DDP(
    model, device_ids=[device_id]
)  # multi-device modules and CPU modules는 device_ids를 None으로 줘야한다고 함.

# 데이터 불러오기
raw_dataset = load_dataset(
    "HAERAE-HUB/KOREAN-SyntheticText-1.5B",
    cache_dir="/data/di-LLM_458/denver.in/LLM/.cache",
)

# 데이터셋 크기가 크므로 10,000개만 샘플링
train_dataset = raw_dataset["train"].shuffle(seed=42).select(range(10000))
if rank == 0:
    print(
        train_dataset[0]
    )  # 단위: 제약 제조 및 기술 - 레이저 절단 기술의 응용\n\n제약 제조 및 기술 분야에서, 레이저 절단 기술은 정밀성과 효율성을 갖춘 혁신적인 제조 공정을 가능하게 하는 중요한 도구입니다...


# 데이터 토크나이징
def tokenize_function(examples):
    return tokenizer(examples["text"], truncation=True, max_length=128)


tokenized_train_dataset = train_dataset.map(
    tokenize_function, batched=True, remove_columns=["text"]
)

# Data Collator 준비
data_collator = DataCollatorForLanguageModeling(tokenizer=tokenizer, mlm=False)

# DistributedSampler를 이용해 train_sampler 정의 (데이터를 각 모델 replica에 나눠 분배)
train_sampler = DistributedSampler(tokenized_train_dataset)

# DataLoader 생성
train_dataloader = DataLoader(
    tokenized_train_dataset,
    batch_size=32,
    sampler=train_sampler,
    collate_fn=data_collator,
)

# 옵티마이저 설정
optimizer = optim.AdamW(ddp_model.parameters(), lr=1e-4)

# 학습 단계
num_epochs = 5
global_step = 0

# 전체 학습 시간 측정 시작
overall_start_time = time.time()

for epoch in range(num_epochs):
    epoch_start_time = time.time()

    train_sampler.set_epoch(epoch)
    ddp_model.train()
    for step, batch in enumerate(train_dataloader):
        input_ids = batch["input_ids"].to(device_id)
        attention_mask = batch["attention_mask"].to(device_id)
        labels = batch["labels"].to(device_id)

        outputs = ddp_model(
            input_ids=input_ids, attention_mask=attention_mask, labels=labels
        )
        loss = outputs.loss

        optimizer.zero_grad()
        loss.backward()
        optimizer.step()

        global_step += 1

        if step % 10 == 0:
            print(
                f"[Rank {rank}] Epoch {epoch} Step {step} " f"Loss: {loss.item():.4f}"
            )

    epoch_end_time = time.time()
    if rank == 0:
        print(
            f"[Rank {rank}] Epoch {epoch} completed in {epoch_end_time - epoch_start_time:.2f} seconds"
        )

# 전체 학습 시간 출력
# 1 GPU: 427.48 seconds
# 2 GPUs: 230.22 seconds
overall_end_time = time.time()
if rank == 0:
    print(f"Training finished in {overall_end_time - overall_start_time:.2f} seconds!")

# 분산 처리 환경 종료
dist.destroy_process_group()

# Inference 단계
if rank == 0:  # inference는 rank 0 에서만 수행
    model.eval()
    prompt = "단위: 제약 제조 및 기술 - 레이저 절단 기술의 응용\n\n"  # 제약 제조 및 기술 분야에서, 레이저 절단 기술은 정밀성과 효율성을 갖춘 혁신적인 제조 공정을 가능하게 하는 중요한 도구입니다...
    input_ids = tokenizer(prompt, return_tensors="pt").input_ids.to(device_id)

    with torch.no_grad():
        generated_ids = model.generate(
            input_ids,
            max_length=50,
            num_return_sequences=1,
            do_sample=False,
        )

    generated_text = tokenizer.decode(generated_ids[0], skip_special_tokens=True)
    print(f"\nGenerated text:\n{generated_text}")

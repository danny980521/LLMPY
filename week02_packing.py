# 본 코드는 https://pytorch.org/tutorials/intermediate/ddp_tutorial.html의 elastic_ddp.py를 기반으로 작성되었습니다.
# torchrun --nproc_per_node=8 /data/di-LLM_458/denver.in/LLMPY/week02_packing.py

## 토치런으로 돌리면 랭크 지정 필요없다?

import os
import time
from contextlib import contextmanager
from typing import Dict, List

import numpy as np
import torch
import torch.distributed as dist
import torch.nn as nn
import torch.optim as optim
from datasets import Features, Sequence, Value, load_dataset
from torch.nn.parallel import DistributedDataParallel as DDP
from torch.utils.data import DataLoader
from torch.utils.data.distributed import DistributedSampler
from transformers import (AutoModelForCausalLM, AutoTokenizer,
                          DataCollatorForLanguageModeling)


@contextmanager
def rank_zero_first(rank):
    """
    Rank 0이 먼저 실행하고, 이후 다른 rank들이 실행하도록 하는 context manager.
    사용 예제: 모델이나 데이터셋 로드를 1개 프로세스만 먼저 실행하고, 나머지 프로세스들은 기다린 후 캐쉬를 이용해 로드
    출처: https://github.com/huggingface/nanotron/blob/9055c664c28a3b430b4e53bfcb5a074068c90f2a/src/nanotron/utils.py#L57-L67
    """

    if rank == 0:
        yield  # Rank 0이 먼저 실행
        dist.barrier()  # Rank 0이 끝날 때까지 다른 Rank들은 대기

    else:
        dist.barrier()  # Rank 0이 끝날 때까지 대기
        yield  # 이후 다른 rank들이 실행


# 현재 프로세스가 사용할 GPU 디바이스 설정 (Local Rank 이용)
torch.cuda.set_device(int(os.environ["LOCAL_RANK"]))

# 분산 처리 환경 초기화 (엔비디아의 NCCL 백엔드 사용)
dist.init_process_group("nccl")

# 현재 프로세스의 rank 확인 (= Global Rank)
rank = dist.get_rank()
print(f"Start running basic DDP example on rank {rank}.")

# 현재 프로세스가 사용할 GPU의 ID 계산
device_id = rank % torch.cuda.device_count()

# 모델 불러오기
model_name = "/data/di-LLM_458/denver.in/LLMPY/models/Llama-3.2-1B"
with rank_zero_first(rank):
    model = AutoModelForCausalLM.from_pretrained(
        model_name,
        # torch_dtype=torch.float32,
        torch_dtype="auto",
        cache_dir="/data/di-LLM_458/denver.in/LLMPY/.cache",
    )

# DDP에 맞추어 모델 이동 & 래핑
# torch.nn.parallel.DistributedDataParallel: 각 model replica에 대해 gradient를 동기화하여 data parallelism을 제공하는 모듈
# 참고: https://pytorch.org/docs/stable/generated/torch.nn.parallel.DistributedDataParallel.html#torch.nn.parallel.DistributedDataParallel
model.to(device_id)
ddp_model = DDP(
    model, device_ids=[device_id]
)  # multi-device modules and CPU modules는 device_ids를 None으로 줘야한다고 함.

# 토크나이저 & 데이터 불러오기
with rank_zero_first(rank):
    tokenizer = AutoTokenizer.from_pretrained(model_name)
    tokenizer.pad_token = tokenizer.eos_token

    raw_dataset = load_dataset(
        "HAERAE-HUB/KOREAN-SyntheticText-1.5B",
        cache_dir="/data/di-LLM_458/denver.in/LLMPY/.cache",
    )

    # 데이터셋 크기가 크므로 10,000개만 샘플링
    train_dataset = raw_dataset["train"].shuffle(seed=42).select(range(10000))
    if rank == 0:
        print(
            train_dataset[0]
        )  # 단위: 제약 제조 및 기술 - 레이저 절단 기술의 응용\n\n제약 제조 및 기술 분야에서, 레이저 절단 기술은 정밀성과 효율성을 갖춘 혁신적인 제조 공정을 가능하게 하는 중요한 도구입니다...

    sequence_length = 4096

    def group_texts(
        examples: Dict[str, List[np.ndarray]],
    ) -> Dict[str, List[np.ndarray]]:
        # 출처: https://github.com/huggingface/nanotron/blob/9055c664c28a3b430b4e53bfcb5a074068c90f2a/src/nanotron/dataloader.py#L292-L307
        # Concatenate all texts.
        concatenated_examples = {k: np.concatenate(v) for k, v in examples.items()}
        total_length = len(concatenated_examples[next(iter(examples.keys()))])
        # WARNING: We drop the small remainder, we could add padding if the model supported it instead of this drop, you can
        # customize this part to your needs.
        if total_length >= sequence_length + 1:
            total_length = ((total_length - 1) // sequence_length) * sequence_length + 1
        # Split by chunks of sequence_length.
        result = {
            k: [
                t[i : i + sequence_length + 1]
                for i in range(0, total_length - (sequence_length + 1), sequence_length)
            ]
            for k, t in concatenated_examples.items()
        }
        return result

    def _tokenize_and_group_texts(texts: List[str]) -> Dict[str, List[np.ndarray]]:
        # 출처: https://github.com/huggingface/nanotron/blob/9055c664c28a3b430b4e53bfcb5a074068c90f2a/src/nanotron/dataloader.py#L309-L312
        tokenized_batch = tokenizer.batch_encode_plus(
            texts, return_attention_mask=False, return_token_type_ids=False
        )
        tokenized_batch = {
            k: [np.array(tokenized_texts) for tokenized_texts in v]
            for k, v in tokenized_batch.items()
        }
        return group_texts(tokenized_batch)

    tokenized_train_dataset = train_dataset.map(
        _tokenize_and_group_texts,
        input_columns="text",
        remove_columns=train_dataset.column_names,
        features=Features(
            {
                "input_ids": Sequence(
                    feature=Value(dtype="int64"), length=sequence_length + 1
                )
            }
        ),
        batched=True,
        num_proc=12,  # 각 프로세스별 서브프로세스 개수
        load_from_cache_file=True,
        desc=f"Grouping texts in chunks of {sequence_length+1}",
    )

    # Data Collator 준비
    data_collator = DataCollatorForLanguageModeling(tokenizer=tokenizer, mlm=False)

    # DistributedSampler를 이용해 train_sampler 정의 (데이터를 각 모델 replica에 나눠 분배)
    train_sampler = DistributedSampler(tokenized_train_dataset)

    # DataLoader 생성
    train_dataloader = DataLoader(
        tokenized_train_dataset,
        batch_size=1,
        sampler=train_sampler,
        collate_fn=data_collator,
    )

# 옵티마이저 설정
optimizer = optim.AdamW(ddp_model.parameters(), lr=1e-4)

# 학습 단계
num_epochs = 1
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

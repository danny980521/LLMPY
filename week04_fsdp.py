# torchrun --nproc_per_node=8 /data/di-LLM_458/denver.in/LLMPY/week04_fsdp.py
# FSDP.ShardingStrategy.SHARD_GRAD_OP를 사용하면 ZeRO Stage 2를 적용 가능

# gradient with bucket view?

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
from torch.cuda.amp import autocast
from torch.distributed.fsdp import BackwardPrefetch, CPUOffload
from torch.distributed.fsdp import FullyShardedDataParallel as FSDP
from torch.distributed.fsdp import MixedPrecision, ShardingStrategy
from torch.distributed.fsdp.wrap import size_based_auto_wrap_policy
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
print(f"Start running FSDP example on rank {rank}.")

# 현재 프로세스가 사용할 GPU의 ID 계산
device_id = rank % torch.cuda.device_count()

# 모델 불러오기
model_name = "/data/di-LLM_458/denver.in/LLMPY/models/Llama-3.2-1B"
with rank_zero_first(rank):
    model = AutoModelForCausalLM.from_pretrained(
        model_name,
        torch_dtype=torch.float32,
        cache_dir="/data/di-LLM_458/denver.in/LLMPY/.cache",
    )

# 모델을 GPU로 이동
model.to(device_id)

# FSDP로 모델 래핑
# 필요 시 아래와 같이 세부 옵션을 설정할 수도 있습니다.
model = FSDP(
    model,
    sharding_strategy=ShardingStrategy.FULL_SHARD,
    cpu_offload=CPUOffload(offload_params=False),
    mixed_precision=MixedPrecision(
        param_dtype=torch.bfloat16,
        reduce_dtype=torch.float32,
        buffer_dtype=torch.bfloat16,
    ),
)
# model = FSDP(model)  # 단순하게 감싸는 버전
# 이후부터는 model이 FSDP로 래핑된 모델이 됩니다.

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
        print(train_dataset[0])

    sequence_length = 8192

    def group_texts(
        examples: Dict[str, List[np.ndarray]],
    ) -> Dict[str, List[np.ndarray]]:
        concatenated_examples = {k: np.concatenate(v) for k, v in examples.items()}
        total_length = len(concatenated_examples[next(iter(examples.keys()))])
        if total_length >= sequence_length + 1:
            total_length = ((total_length - 1) // sequence_length) * sequence_length + 1
        result = {
            k: [
                t[i : i + sequence_length + 1]
                for i in range(0, total_length - (sequence_length + 1), sequence_length)
            ]
            for k, t in concatenated_examples.items()
        }
        return result

    def _tokenize_and_group_texts(texts: List[str]) -> Dict[str, List[np.ndarray]]:
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

# bf16 mixed precision 스칼라(autocast) 설정
# 필요 시 torch.bfloat16 대신 torch.float16 등 원하는 dtype으로 조정 가능
# from torch.cuda.amp import autocast  # 이미 import 되어 있음

# Optimizer 설정 (FSDP로 이미 파라미터가 sharded 되므로 일반 AdamW 사용 가능)
optimizer = optim.AdamW(model.parameters(), lr=5e-5)

# 학습 단계
num_epochs = 1
gradient_accumulation_steps = 2
global_step = 0

# 전체 학습 시간 측정 시작
overall_start_time = time.time()

for epoch in range(num_epochs):
    epoch_start_time = time.time()

    train_sampler.set_epoch(epoch)
    model.train()
    optimizer.zero_grad()

    for step, batch in enumerate(train_dataloader):
        input_ids = batch["input_ids"].to(device_id)
        attention_mask = batch["attention_mask"].to(device_id)
        labels = batch["labels"].to(device_id)

        # # bf16 mixed precision 사용
        # with autocast(dtype=torch.bfloat16):
        outputs = model(
            input_ids=input_ids, attention_mask=attention_mask, labels=labels
        )
        # Scale the loss by 1/gradient_accumulation_steps
        loss = outputs.loss / gradient_accumulation_steps

        # Backprop
        loss.backward()

        # Only step optimizer after accumulating enough gradients
        if (step + 1) % gradient_accumulation_steps == 0:
            optimizer.step()
            optimizer.zero_grad()

        # For logging, you might want the *unscaled* loss here
        if step % 10 == 0:
            # Multiply back by grad_accumulation_steps if you want the original scale:
            current_loss = loss.item() * gradient_accumulation_steps
            print(f"[Rank {rank}] Epoch {epoch} Step {step} Loss: {current_loss:.4f}")

        global_step += 1

    epoch_end_time = time.time()
    if rank == 0:
        print(
            f"[Rank {rank}] Epoch {epoch} completed in {epoch_end_time - epoch_start_time:.2f} seconds"
        )

# 전체 학습 시간 출력
overall_end_time = time.time()
if rank == 0:
    print(f"Training finished in {overall_end_time - overall_start_time:.2f} seconds!")

# FSDP 샤딩 해제 과정
from torch.distributed.fsdp import FullStateDictConfig, StateDictType

# FSDP model로부터 full state dict를 가져올 때 사용하는 config
full_sd_config = FullStateDictConfig(offload_to_cpu=True, rank0_only=True)

# 아래 컨텍스트 안에서 model.state_dict()를 호출하면,
# 모든 rank의 shard가 자동으로 모여 rank 0에만 전체 state_dict가 생성됩니다.
with FSDP.state_dict_type(model, StateDictType.FULL_STATE_DICT, full_sd_config):
    full_sd = model.state_dict()

# rank 0에서만 전체 state_dict를 가지고, 동일 아키텍처의 일반 모델을 다시 생성
if rank == 0:
    # 원래 모델과 동일한 구조로 로딩 (가중치는 load_state_dict로 대체할 예정)
    normal_model = AutoModelForCausalLM.from_pretrained(
        model_name,
        torch_dtype=torch.float32,
        cache_dir="/data/di-LLM_458/denver.in/LLMPY/.cache",
    )
    # FSDP 없이도 쓸 수 있도록 full_sd를 로드
    normal_model.load_state_dict(full_sd, strict=True)
    normal_model.to(device_id)
    # inference에서는 이 모델을 사용
    model = normal_model


# Inference 단계
if rank == 0:  # inference는 rank 0 에서만 수행
    model.eval()
    prompt = "단위: 제약 제조 및 기술 - 레이저 절단 기술의 응용\n\n"
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


# 분산 처리 환경 종료
dist.destroy_process_group()

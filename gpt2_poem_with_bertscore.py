# coding=utf-8
# Copyright 2022 The HuggingFace Inc. team. All rights reserved.
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#     http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.
import torch
from tqdm import tqdm
from transformers import AutoTokenizer, pipeline, AutoModelForSequenceClassification

from trl import AutoModelForCausalLMWithValueHead, PPOConfig, PPOTrainer, set_seed
from trl.core import LengthSampler
from evaluate import load
import random 
import time
from iTrainingLogger import iSummaryWriter
from utils import load_prompts

tqdm.pandas()

########################################################################
# This is a fully working simple example to use trl with accelerate.
#
# This example fine-tunes a GPT2 model on the IMDB dataset using PPO
# (proximal policy optimization).
# in any of the following settings (with the same script):
#   - single CPU or single GPU
#   - multi GPUS (using PyTorch distributed mode)
#   - multi GPUS (using DeepSpeed ZeRO-Offload stages 1 & 2)
#   - fp16 (mixed-precision) or fp32 (normal precision)
#
# To run it in each of these various modes, first initialize the accelerate
# configuration with `accelerate config`
#
########################################################################

# We first define the configuration of the experiment, defining the model, the dataset,
# the training parameters, and the PPO parameters.
# Check the default arguments in the `PPOConfig` class for more details.
# If you want to log with tensorboard, add the kwarg
# `accelerator_kwargs={"logging_dir": PATH_TO_LOGS}` to the PPOConfig.

writer = iSummaryWriter(log_path='./logs', log_name='PPO-Sentiment-gpt')
config = PPOConfig(
    model_name="ismaelfaro/gpt2-poems.en",
    learning_rate=1.41e-5,
    batch_size=64,
    forward_batch_size=16, # could be increased
)
# prompt池
# prompts = [
#     'When I was fair and young,',
#     'Let the bird of loudest',
#     'It were enough that',
#     'Come live with me',
#     'See the chariot at hand here of Love,'
# ]
prompts = load_prompts(5)[:5]

# We then define the arguments to pass to the sentiment analysis pipeline.
# We set `return_all_scores` to True to get the sentiment score for each token.
sent_kwargs = {"return_all_scores": True, "function_to_apply": "none", "batch_size": 16}



# set seed before initializing value head for deterministic eval
set_seed(config.seed)

# Now let's build the model, the reference model, and the tokenizer.
model = AutoModelForCausalLMWithValueHead.from_pretrained(config.model_name)
ref_model = AutoModelForCausalLMWithValueHead.from_pretrained(config.model_name)
tokenizer = AutoTokenizer.from_pretrained(config.model_name)

# GPT-2 tokenizer has a pad token, but it is not eos_token by default. We need to set it to eos_token.
# only for this model.
tokenizer.pad_token = tokenizer.eos_token

# We then build the PPOTrainer, passing the model, the reference model, the tokenizer
ppo_trainer = PPOTrainer(config, model, ref_model, tokenizer)
ppo_trainer_ref = PPOTrainer(config, ref_model, ref_model, tokenizer)
# We then build the sentiment analysis pipeline, passing the model name and the
# sentiment analysis pipeline arguments. Let's also make sure to set the device
# to the same device as the PPOTrainer.
device = ppo_trainer.accelerator.device
if ppo_trainer.accelerator.num_processes == 1:
    device = 0 if torch.cuda.is_available() else "cpu"  # to avoid a `pipeline` bug
sentiment_pipe = pipeline("sentiment-analysis", model="nickwong64/bert-base-uncased-poems-sentiment", device=device)

# bleu_tokenizer = AutoTokenizer.from_pretrained("Elron/bleurt-tiny-512")
# bleu_model = AutoModelForSequenceClassification.from_pretrained("Elron/bleurt-tiny-512")
# bleu_model.eval()
bertscore = load("bertscore")
# We then define the arguments to pass to the `generate` function. These arguments
# are passed to the `generate` function of the PPOTrainer, which is a wrapper around
# the `generate` function of the trained model.
generation_kwargs = {
    "min_length": -1,
    "top_k": 0.0,
    "top_p": 1.0,
    "do_sample": True,
    "pad_token_id": tokenizer.eos_token_id,
}
output_min_length = 4
output_max_length = 128 # Increase the max length
output_length_sampler = LengthSampler(output_min_length, output_max_length)


for epoch in tqdm(range(500)):
    logs, timing = dict(), dict()
    t0 = time.time()
    batch = {
        'tokens': [],
        'query': []
    }
    for _ in range(config.batch_size):
        random_prompt = random.choice(prompts)                                  # 随机选择一个prompt
        tokens = tokenizer.encode(random_prompt)
        batch['tokens'].append(tokens)
        batch['query'].append(random_prompt)
    query_tensors = [torch.tensor(t).long().to(device) for t in batch["tokens"]]
    
    t = time.time()
     
    response_tensors = []
    ref_response_tensors = []
    for query in query_tensors:
        gen_len = output_length_sampler()
        generation_kwargs["max_new_tokens"] = gen_len
        response = ppo_trainer.generate(query, **generation_kwargs)
        ref_response = ppo_trainer_ref.generate(query, **generation_kwargs)
        response_tensors.append(response.squeeze()[-gen_len:])
        ref_response_tensors.append(ref_response.squeeze()[-gen_len:])
    batch["response"] = [tokenizer.decode(r.squeeze()) for r in response_tensors]
    batch["ref_response"] = [tokenizer.decode(r.squeeze()) for r in ref_response_tensors]
    
    # response_tensors = []
    # for query in query_tensors:
    #     gen_len = output_length_sampler()
    #     generation_kwargs["max_new_tokens"] = gen_len
    #     response = ppo_trainer.generate(query, **generation_kwargs)
    #     response_tensors.append(response.squeeze())
    # batch["response"] = [tokenizer.decode(r[1:].squeeze()) for r in response_tensors]
    
    t = time.time()
    texts = [q + r for q, r in zip(batch["query"], batch["response"])]
    pipe_outputs = sentiment_pipe(texts)
    # Compute bert score
    predictions = batch["response"]
    references = batch["ref_response"]
    results = bertscore.compute(predictions=predictions, references=references, model_type="distilbert-base-uncased")
    results_bert = results['precision']

    rewards = []

    for idx,output in enumerate(pipe_outputs):
        # print(output)
        if output['label'] == 'positive':
            rewards.append(torch.tensor(output['score']+results_bert[idx]).to(device))
        elif output['label'] == 'no_impact' or output['label'] == 'mixed':
            rewards.append(torch.tensor(0.0+results_bert[idx]).to(device))
        elif output['label'] == 'negative':
            rewards.append(torch.tensor(-output['score']+results_bert[idx]).to(device))
        else:
            raise ValueError(f"Wrong {output['label']}.")
    # rewards = torch.tensor(rewards).to(device)                                  # 将正向情感的得分作为生成得分
    # rewards = [torch.tensor(output[1]["score"]).to(device) for output in pipe_outputs]
    print(rewards)

    timing['time/get_sentiment_preds'] = time.time() - t
    # Run PPO step
    stats = ppo_trainer.step(query_tensors, response_tensors, rewards)
    # ppo_trainer.log_stats(stats, batch, rewards)
    timing['time/optimization'] = time.time() - t

    timing['time/epoch'] = time.time() - t0                                     # logging
    logs.update(timing)
    logs.update(stats)
    
    logs['env/reward_mean'] = (sum(rewards) / len(rewards)).cpu()
    # logs['env/reward_std'] = torch.std(rewards).cpu().numpy()
    # logs['env/reward_dist'] = rewards.cpu().numpy()
    print(f"epoch {epoch} mean-reward: {logs['env/reward_mean']}")
    
    # print('Random Sample 5 text(s) of model output:')
    # for i in range(5):                                                           # 随机打5个生成的结果
        # print(f'{i+1}. {random.choice(texts)}')
    model.save_pretrained('gpt2-poem', push_to_hub=False)
    tokenizer.save_pretrained('gpt2-poem', push_to_hub=False)
    writer.add_scalar('train/reward', logs['env/reward_mean'], epoch)
    for k, v in timing.items():
        writer.add_scalar(k, v, epoch)
    writer.add_scalar("objective/entropy", stats["objective/entropy"], epoch)
    writer.add_scalar("objective/kl", stats["objective/kl"], epoch)
    # writer.add_scalar("objective/logprobs", stats["objective/logprobs"], epoch)
    # writer.add_scalar("ppo/mean_non_score_reward", stats["ppo/mean_non_score_reward"], epoch)
    writer.record()
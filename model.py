import json
import urllib3
import pathlib
import shutil
import requests
import os
import re
import random

ALLOW_NEW_LINES = False    
LEARNING_RATE = 1.372e-4
EPOCHS = 4

#adapted from https://github.com/borisdayma/huggingtweets/blob/master/huggingtweets-demo.ipynb
def fix_text(text):
    text = text.replace('&amp;', '&')
    text = text.replace('&lt;', '<')
    text = text.replace('&gt;', '>')
    return text

def clean_tweet(tweet, allow_new_lines = ALLOW_NEW_LINES):
    bad_start = ['http:', 'https:']
    for w in bad_start:
        tweet = re.sub(f" {w}\\S+", "", tweet)      # removes white space before url
        tweet = re.sub(f"{w}\\S+ ", "", tweet)      # in case a tweet starts with a url
        tweet = re.sub(f"\n{w}\\S+ ", "", tweet)    # in case the url is on a new line
        tweet = re.sub(f"\n{w}\\S+", "", tweet)     # in case the url is alone on a new line
        tweet = re.sub(f"{w}\\S+", "", tweet)       # any other case?
    tweet = re.sub(' +', ' ', tweet)                # replace multiple spaces with one space
    if not allow_new_lines:                         # TODO: predictions seem better without new lines
        tweet = ' '.join(tweet.split())
    return tweet.strip()

def boring_tweet(tweet):
    "Check if this is a boring tweet"
    boring_stuff = ['http', '@', '#']
    not_boring_words = len([None for w in tweet.split() if all(bs not in w.lower() for bs in boring_stuff)])
    return not_boring_words < 3

def get_data(handle):
    http = urllib3.PoolManager(retries=urllib3.Retry(3))
    res = http.request("GET", f"http://us-central1-huggingtweets.cloudfunctions.net/get_tweets?handle={handle}&force=1")
    res = json.loads(res.data.decode('utf-8'))

    all_tweets = res['tweets']
    curated_tweets = [fix_text(tweet) for tweet in all_tweets]

    # create dataset
    clean_tweets = [clean_tweet(tweet) for tweet in curated_tweets]
    cool_tweets = [tweet for tweet in clean_tweets if not boring_tweet(tweet)]

    # create a file based on multiple epochs with tweets mixed up
    seed_data = random.randint(0,2**32-1)
    dataRandom = random.Random(seed_data)
    total_text = '<|endoftext|>'
    all_handle_tweets = []
    epoch_len = max(len(''.join(cool_tweet)) for cool_tweet in cool_tweets)
    for _ in range(EPOCHS):
        for cool_tweet in cool_tweets:
            dataRandom.shuffle(cool_tweet)
            current_tweet = cool_tweet
            current_len = len(''.join(current_tweet))
            while current_len < epoch_len:
                for t in cool_tweet:
                    current_tweet.append(t)
                    current_len += len(t)
                    if current_len >= epoch_len: break
            dataRandom.shuffle(current_tweet)
            all_handle_tweets.extend(current_tweet)
    total_text += '<|endoftext|>'.join(all_handle_tweets) + '<|endoftext|>'
    with open(f"data_{'-'.join(handle)}_train.txt", 'w') as f:
        f.write(total_text)

def finetune(handle):
    # transformers imports later as wandb needs to have logged in
    import transformers
    from transformers import (
        AutoTokenizer, AutoModelForCausalLM,
        TextDataset, DataCollatorForLanguageModeling,
        Trainer, TrainingArguments,
        get_cosine_schedule_with_warmup)

    # Setting up pre-trained neural network
    global trainer
    tokenizer = AutoTokenizer.from_pretrained('gpt3')
    model = AutoModelForCausalLM.from_pretrained('gpt3', cache_dir=pathlib.Path('cache').resolve())
    block_size = tokenizer.model_max_length
    train_dataset = TextDataset(tokenizer=tokenizer, file_path=f"data_{'-'.join(handle)}_train.txt", block_size=block_size, overwrite_cache=True)
    data_collator = DataCollatorForLanguageModeling(tokenizer=tokenizer, mlm=False)
    seed = random.randint(0,2**32-1)

    training_args = TrainingArguments(
        output_dir=f"output/{'-'.join(handle)}",
        overwrite_output_dir=True,
        do_train=True,
        num_train_epochs=1,
        per_device_train_batch_size=1,
        prediction_loss_only=True,
        logging_steps=5,
        save_steps=0,
        seed=seed,
        learning_rate = LEARNING_RATE)

    trainer = Trainer(
                    model=model,
                    tokenizer=tokenizer,
                    args=training_args,
                    data_collator=data_collator,
                    train_dataset=train_dataset)
                
    # Update lr scheduler
    train_dataloader = trainer.get_train_dataloader()
    num_train_steps = len(train_dataloader)
    trainer.create_optimizer_and_scheduler(num_train_steps)
    trainer.lr_scheduler = get_cosine_schedule_with_warmup(
        trainer.optimizer,
        num_warmup_steps=0,
        num_training_steps=num_train_steps)

    trainer.train()

    trainer.model.config.task_specific_params['text-generation'] = {
        'do_sample': True,
        'min_length': 10,
        'max_length': 160,
        'temperature': 1.,
        'top_p': 0.95,
        'prefix': '<|endoftext|>'}

    model_name = '-'.join(handle)
    trainer.save_model(model_name)
    return trainer 

def predict(start, trainer):
    start_with_bos = '<|endoftext|>' + start
    encoded_prompt = trainer.tokenizer(start_with_bos, add_special_tokens=False, return_tensors="pt").input_ids
    encoded_prompt = encoded_prompt.to(trainer.model.device)

    # prediction
    output_sequences = trainer.model.generate(
        input_ids=encoded_prompt,
        max_length=160,
        min_length=10,
        temperature=1.,
        top_p=0.95,
        do_sample=True,
        num_return_sequences=10
        )
    generated_sequences = []

    # decode prediction
    for generated_sequence_idx, generated_sequence in enumerate(output_sequences):
        generated_sequence = generated_sequence.tolist()
        text = trainer.tokenizer.decode(generated_sequence, clean_up_tokenization_spaces=True, skip_special_tokens=True)
        if not ALLOW_NEW_LINES:
            limit = text.find('\n')
            text = text[: limit if limit != -1 else None]
        generated_sequences.append(text.strip())

    predictions = []
    for i, g in enumerate(generated_sequences):
        predictions.append([start, g])
    return predictions

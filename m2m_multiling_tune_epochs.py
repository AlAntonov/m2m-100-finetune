#!/usr/bin/env python

import sys

import json

from random import shuffle

from transformers import AutoModelForSeq2SeqLM, DataCollatorForSeq2Seq, Seq2SeqTrainingArguments, Seq2SeqTrainer, M2M100Tokenizer

from datasets import load_dataset, load_metric

from datetime import datetime

def log(msg):
	print(str(datetime.now()) + ": " + str(msg))

def get_trainer(tok, mdl, trainset, devset, devmeta, outdir, batch_size = 1, gradient_accumulation_steps = 4, learning_rate = 5e-05, weight_decay = 0.00, num_epochs = 10):
	args = Seq2SeqTrainingArguments(
		 outdir,
		 evaluation_strategy = "epoch",
		 save_strategy = "epoch",
		 learning_rate=learning_rate,
		 per_device_train_batch_size=batch_size,
		 per_device_eval_batch_size=batch_size,
		 weight_decay=weight_decay,
		 gradient_accumulation_steps=gradient_accumulation_steps,
		 save_total_limit=None,
		 num_train_epochs=num_epochs,
		 predict_with_generate=True,
                 logging_dir='logs'   
	)
	
	data_collator = DataCollatorForSeq2Seq(tok, model=mdl)
	
	metric = load_metric("sacrebleu")
	
	def compute_metrics(eval_preds):
		hyp, ref = eval_preds
		if isinstance(hyp, tuple):
			hyp = hyp[0]
		
		dechyp = [pr.strip() for pr in tok.batch_decode(hyp, skip_special_tokens=True)]
		decref = [[hp.strip()] for hp in tok.batch_decode(ref, skip_special_tokens=True)]
		
		currStart = 0
		result = {}
		for filename, rownum in devmeta:
			metrresult = metric.compute(predictions=dechyp[currStart:currStart+rownum], references=decref[currStart:currStart+rownum])
			keyname = "bleu_" + filename
			result[keyname] = metrresult['score']
			currStart += rownum
		
		return result
	
	return Seq2SeqTrainer(
		mdl,
		args,
		train_dataset=trainset,
		eval_dataset=devset,
		data_collator=data_collator,
		tokenizer=tok,
		compute_metrics=compute_metrics
	)

def loadmdl(initmdl, newnum):
	result = AutoModelForSeq2SeqLM.from_pretrained(initmdl)
	
	result.resize_token_embeddings(newnum)
	
	return result

if __name__ == "__main__":
	_, outdir = sys.argv

	log("Load model")         	
	tokenizer = M2M100Tokenizer.from_pretrained("facebook/m2m100_418M")
	model = loadmdl("facebook/m2m100_418M", len(tokenizer))
	
	log("Load dataset")
	data = load_dataset('masakhane/mafand', 'en-hau')

	small_train_data = data['train'].shuffle(seed=42).select(range(1000))
	devlen = 100
	small_test_data = data['validation'].shuffle(seed=42).select(range(devlen))

	tokenizer.src_lang = 'ha'
	tokenizer.tgt_lang = 'en'
	
	def tokenize_function(examples):
		ins = [ex['hau'] for ex in examples['translation']]
		outs = [ex['en'] for ex in examples['translation']]
		
		result = tokenizer(ins, max_length=128, padding=True, truncation=True)
		
		with tokenizer.as_target_tokenizer():
			labels = tokenizer(outs, max_length=128, padding=True, truncation=True)
		
		result['labels'] = labels['input_ids']
		
		return result

	traindata = small_train_data.map(tokenize_function, batched=True, desc="tokenize_function", remove_columns=['translation'])
	devdata = small_test_data.map(tokenize_function, batched=True, desc="tokenize_function files", remove_columns=['translation'])
	
	log("Start training")
	devmeta = [('file_dev', devlen)]
	for filename, rownum in devmeta:
		print(filename)
		print(rownum)
	trainer = get_trainer(tokenizer, model, traindata, devdata, devmeta, outdir, num_epochs = 1)
	
	log("Starting training")
	trainer.train()

	log("Saving model")
	trainer.save_model(outdir)
	
	log("Done!")

from transformers import AutoTokenizer, M2M100ForConditionalGeneration

base_model = "facebook/m2m100_418M"
local_model = "/content/checkpoint-2"
model = M2M100ForConditionalGeneration.from_pretrained(base_model)
tokenizer = AutoTokenizer.from_pretrained(base_model)

text_to_translate = "Wannan dai shi ne karo na 3 da birnin Landan ke karɓar baƙuncin gasar wasanni ta ƙasa da ƙasa da dama."
model_inputs = tokenizer(text_to_translate, return_tensors="pt")

# translate
gen_tokens = model.generate(**model_inputs, forced_bos_token_id=tokenizer.get_lang_id("en"))
print(tokenizer.batch_decode(gen_tokens, skip_special_tokens=True))
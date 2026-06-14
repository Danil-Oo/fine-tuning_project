import os

from dotenv import load_dotenv
from unsloth import FastLanguageModel

load_dotenv()

LORA_PATH = os.getenv("LORA_PATH", "./adapters/run_01/final_adapter")
OUTPUT_PATH = os.getenv("MERGED_MODEL_PATH", "./models/merged")

print(f"Загружаем модель + адаптеры из: {LORA_PATH}")
model, tokenizer = FastLanguageModel.from_pretrained(
    model_name=LORA_PATH,  # unsloth сам прочитает adapter_config.json
    max_seq_length=32768,
    load_in_4bit=True,
)

print(f"Мержим и сохраняем в: {OUTPUT_PATH}")
model.save_pretrained_merged(
    OUTPUT_PATH,
    tokenizer,
    save_method="merged_16bit",
)

print("Готово!")

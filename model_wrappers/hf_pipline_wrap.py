
from transformers import pipeline, logging
import torch

logging.set_verbosity_info()


class HfPipelineWrap:
    max_new_tokens = 500

    def __init__(self, model_name, temperature, batch_size, torch_dtype=torch.bfloat16, load_in_4_bit=False, load_in_8_bit=False):
        self.model_name = model_name
        print(f"loading {model_name}")
        # model_kwargs = {"torch_dtype": torch_dtype, "device_map": "auto",
        #                 "load_in_4bit": load_in_4_bit, "load_in_8bit": load_in_8_bit, "attn_implementation":"flash_attention_2"}
        model_kwargs = {"torch_dtype": torch_dtype, "device_map": "auto",
                        "load_in_4bit": load_in_4_bit, "load_in_8bit": load_in_8_bit}

        print(model_kwargs)
        self.pipe = pipeline("text-generation", model=model_name, model_kwargs=model_kwargs, batch_size=batch_size)
        self.temperature = temperature

        if model_name != "meta-llama/Llama-3.1-8B-Instruct":
            self.pipe.tokenizer.padding_side = "left"
            # print(self.pipe.model.config.eos_token_id)
            self.pipe.tokenizer.pad_token_id = self.pipe.model.config.eos_token_id
        else:
            # self.terminators = [
            #     self.pipe.tokenizer.eos_token_id,
            #     self.pipe.tokenizer.convert_tokens_to_ids("<|eot_id|>")
            # ]
            # pipeline.tokenizer.pad_token_id = pipeline.model.config.eos_token_id[0]
            self.pipe.tokenizer.pad_token_id = self.pipe.model.config.eos_token_id[0]
        # print(self.pipe.model)

    def get_max_window(self):
        return self.pipe.model.config.max_position_embeddings

    def batch(self, prompts, num_truncation_tokens):
        encoded_input = [self.pipe.tokenizer.apply_chat_template(
            p, truncation=True, max_length=num_truncation_tokens-self.max_new_tokens)
            for p in prompts]
        if not self.model_name == "google/gemma-2-9b-it":
            decoded_input = self.pipe.tokenizer.batch_decode(encoded_input, skip_special_tokens=True)
        else:
            decoded_input = prompts

        if self.model_name != "meta-llama/Llama-3.1-8B-Instruct":
            outputs = self.pipe(
                decoded_input,
                temperature=self.temperature,
                do_sample=True,
                pad_token_id=self.pipe.tokenizer.eos_token_id,
                max_new_tokens=self.max_new_tokens,
                num_workers = 5
            )
        else:
            outputs = self.pipe(
                decoded_input,
                temperature=self.temperature,
                do_sample=True,
                pad_token_id=self.pipe.tokenizer.eos_token_id,
                max_new_tokens=self.max_new_tokens,
                num_workers=5
            )



        only_responses = [output[0]["generated_text"] for i, output in enumerate(outputs)]
        return only_responses



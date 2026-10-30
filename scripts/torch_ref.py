"""PyTorch reference: greedy generation + logits dump for LFM2.5-350M."""
import json, torch
from transformers import AutoModelForCausalLM, AutoTokenizer

D = "/home/z/my-project/olai/artifacts/patient/LFM2.5-350M/original"
tok = AutoTokenizer.from_pretrained(D)
model = AutoModelForCausalLM.from_pretrained(D, dtype=torch.bfloat16)
model.eval()

prompt = "<|startoftext|><|im_start|>user\nHello! What is 2+2? Answer in one short sentence.<|im_end|>\n<|im_start|>assistant\n"
ids = tok.encode(prompt, add_special_tokens=False)
print("prompt ids:", ids)

with torch.no_grad():
    # full logits at last prompt position (for numeric diff)
    logits = model(input_ids=torch.tensor([ids])).logits[0, -1].float()
    ref = {
        "ids": ids,
        "argmax": int(logits.argmax()),
        "logits_head": logits[:4096].tolist(),
        "logit_at_1024": float(logits[1024]),
        "gen_ids": model.generate(
            torch.tensor([ids]),
            max_new_tokens=24,
            do_sample=False,
            num_beams=1,
            temperature=None, top_p=None, top_k=None,
            pad_token_id=0,
        )[0].tolist()[len(ids):],
    }
json.dump(ref, open("/tmp/torch_ref.json", "w"))
print("argmax:", ref["argmax"], "gen:", ref["gen_ids"])

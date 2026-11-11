#!/usr/bin/env python3
"""NaVIDA Qwen multimodal inference server for hybrid eval (sm_120 torch)."""
import argparse, base64, io, json, os, traceback
from http.server import BaseHTTPRequestHandler, HTTPServer
import torch
from PIL import Image
from transformers import AutoModelForImageTextToText, AutoProcessor
from qwen_vl_utils import process_vision_info

MODEL = None
PROCESSOR = None
DEVICE = "cuda"
MODEL_PATH = None

def load_model(path):
    global MODEL, PROCESSOR, MODEL_PATH
    MODEL_PATH = str(path)
    torch.manual_seed(41)
    kwargs = dict(attn_implementation="flash_attention_2", torch_dtype=torch.bfloat16)
    MODEL = AutoModelForImageTextToText.from_pretrained(path, **kwargs)
    MODEL.config.use_cache = True
    MODEL.to(DEVICE).eval()
    PROCESSOR = AutoProcessor.from_pretrained(path, use_fast=False)
    print("MODEL_LOADED", path, "attention=", MODEL.config._attn_implementation, flush=True)

def b64_to_pil(s):
    return Image.open(io.BytesIO(base64.b64decode(s))).convert("RGB")

@torch.inference_mode()
def generate(payload):
    # payload: {messages: [...]} same structure as HF chat with image_url data URIs OR images_b64 list + text
    messages = payload["messages"]
    text = PROCESSOR.apply_chat_template(messages, tokenize=False, add_generation_prompt=True)
    image_inputs, video_inputs = process_vision_info(messages)
    inputs = PROCESSOR(text=[text], images=image_inputs, videos=video_inputs, padding=True, return_tensors="pt")
    inputs = inputs.to(DEVICE)
    out = MODEL.generate(**inputs, max_new_tokens=int(payload.get("max_new_tokens", 128)), do_sample=False, temperature=None, top_p=None, top_k=None, repetition_penalty=1.05, use_cache=True)
    trimmed = [o[len(i):] for i, o in zip(inputs.input_ids, out)]
    text_out = PROCESSOR.batch_decode(trimmed, skip_special_tokens=True, clean_up_tokenization_spaces=False)[0]
    return {"text": text_out}

class Handler(BaseHTTPRequestHandler):
    def log_message(self, fmt, *args):
        print("[%s] %s" % (self.log_date_time_string(), fmt % args), flush=True)
    def do_GET(self):
        if self.path == "/health":
            self.send_response(200); self.end_headers(); self.wfile.write(json.dumps({"pid": os.getpid(), "model_path": MODEL_PATH, "attention": MODEL.config._attn_implementation}).encode()); return
        self.send_response(404); self.end_headers()
    def do_POST(self):
        if self.path != "/generate":
            self.send_response(404); self.end_headers(); return
        n = int(self.headers.get("Content-Length", 0))
        body = json.loads(self.rfile.read(n))
        try:
            res = generate(body)
            data = json.dumps(res).encode()
            self.send_response(200)
            self.send_header("Content-Type", "application/json")
            self.send_header("Content-Length", str(len(data)))
            self.end_headers(); self.wfile.write(data)
        except Exception as e:
            traceback.print_exc()
            data = json.dumps({"error": str(e)}).encode()
            self.send_response(500)
            self.send_header("Content-Type", "application/json")
            self.send_header("Content-Length", str(len(data)))
            self.end_headers(); self.wfile.write(data)

def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--model-path", required=True)
    ap.add_argument("--host", default="127.0.0.1")
    ap.add_argument("--port", type=int, default=8765)
    ap.add_argument("--gpu", default="0")
    args = ap.parse_args()
    os.environ["CUDA_VISIBLE_DEVICES"] = args.gpu
    load_model(args.model_path)
    print(f"SERVING http://{args.host}:{args.port}", flush=True)
    HTTPServer((args.host, args.port), Handler).serve_forever()

if __name__ == "__main__":
    main()

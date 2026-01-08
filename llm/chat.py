#!/usr/bin/env python3

from argparse import ArgumentParser
from time import time
from icecream import ic
from dba.ic_utl import IcUtl
from llama_cpp import Llama
from dba.os import Os
from os import cpu_count
from types import SimpleNamespace
from os import stat
import sys
from pynvml import (
    nvmlInit,
    nvmlDeviceGetHandleByIndex,
    nvmlDeviceGetMemoryInfo,
    nvmlShutdown,
)
from textwrap import dedent

nvmlInit()
_gpu = nvmlDeviceGetHandleByIndex(0)



class Chat:


    def __init__(self):
        self.args = None

    def pars_args(self):
        ap = ArgumentParser(description='Description of this python script')
        ap.add_argument('--debug', '-d', action='store_true')
        ap.add_argument('--model-file', '--gguf', '-f', required=True)
        ap.add_argument('--verbose-load-model', action='store_true', default=False)
        # ap.add_argument('--verbose-inference', action='store_true', default=False)
        self.args = ap.parse_args()
        IcUtl.debug(self.args)

    def load_model(self):

        stderr_orig = sys.stderr
        with open('llama.load.log', 'w') as f:
            st = stat(self.args.model_file)
            file_gb = round(st.st_size/1024**3, 1)
            ic(self.args.model_file, file_gb)

            if ic.enabled:
                Os.run_free()

            llama_kwargs = dict(
                model_path=self.args.model_file,
                # n_ctx=128,
                n_ctx=2048,
                # max_tokens = 4096,
                # max_tokens=128,
                n_threads=8,          # Try 8 first (common sweet spot); adjust to your physical cores
                n_threads_batch=8,    # Match above
                #n _batch=256,         # Higher = much faster generation on CPU
                n_batch=128,         # Higher = much faster generation on CPU
                n_gpu_layers=-1,      
                use_mlock=False,       # Lock in RAM if you have headroom (prevents swapping)
                verbose=self.args.verbose_load_model
            )
            ic(llama_kwargs)

            ic("loading model . . .")
            time0 = time()
            sys.stderr = f
            llm = Llama(
                **(ic(llama_kwargs))
            )
            sys.stderr = stderr_orig
            ic(". . . done loading model. elapsed %s s" % round(time() - time0))
            ic(cpu_count())
            self.ic_memory_info()
            self.ic_vram_info()


        return llm

    @staticmethod
    def prompt(resp_hist, user_input):

        resp_hist.append({"role": "user", "content": user_input})

        full_prompt = ""
        for msg in resp_hist:
            if msg["role"] == "system":
                full_prompt += f"<start_of_turn>system\n{msg['content']}<end_of_turn>\n"
            elif msg["role"] == "user":
                full_prompt += f"<start_of_turn>user\n{msg['content']}<end_of_turn>\n"
            elif msg["role"] == "model":
                full_prompt += f"<start_of_turn>model\n{msg['content']}<end_of_turn>\n"

        full_prompt += "<start_of_turn>model\n"  # Start the model's turn
        return full_prompt

    def chat(self, llm):
        print("End your multi-line message with Ctrl-D")
        print("Enter ctrl-D to break the loop or exit.")
        """
        1) If you are not certain, reply with exactly: "I don't know."
        """

        system_rules = dedent("""
        You are a factual assistant.
        You must follow these rules:
        1) Report only facts that have evidence.
        2) If ungrounded, reply with exactly: "I don't know."
        3) Do not cite sources or URLs.
        4) No chit-chat. Get to the point.
        5) Output ONLY the final answer. No explanation. No reasoning. No preamble.
        """)

        resp_hist = [
            {
                "role": "system",
                "content": system_rules
            }
        ]

        llm_kwargs = dict(
            # max_tokens=64,
            max_tokens=8192,
            temperature=0.7,
            # temperature=0.0,
            stop=["<end_of_turn>", "</think>"],
            stream=True,
            # verbose=self.args.verbose_inference
        )
        ic(llm_kwargs)

        while True:

            user_input = ''
            while True:
                try:
                    user_input += input("You: ") + '\n'
                    if user_input.lower().strip() in ["/nodebug"]:
                        ic.disable()
                except EOFError as e:
                    break
            ic(user_input)

            full_prompt = self.prompt(resp_hist, user_input)
            ic(full_prompt)

            time0 = time()
            assistant_reply = ""

            try:
                ic("begin inference . . .")
                response = llm(
                    full_prompt,
                    **llm_kwargs
                )
                print("\nAssistant... ", end="", flush=True)
                ellips = ' ...'
                last_reason = None
                for chunk in response:
                    for choice in chunk["choices"]:
                        text = choice["text"]
                        last_reason =  choice['finish_reason']
                        print(ellips + text, end="", flush=True)
                        ellips = ''
                        assistant_reply += text
                print("\n")
                ic(". . . streaming inference done. elapsed %s s" % round(time() - time0))
                ic(last_reason)
                ic(len(assistant_reply))
                self.ic_memory_info()
                self.ic_vram_info()
            except KeyboardInterrupt as e:
                print()

            # Add assistant reply to resp_hist
            resp_hist.append({"role": "model", "content": assistant_reply.strip()})

    @staticmethod
    def ic_memory_info():
        mi = Os.getpid_memoryinfo()
        mg = SimpleNamespace(**mi._asdict())
        for atr in mg.__dict__:
            value = getattr(mg, atr)
            setattr(mg, atr, round(value / 1024**3, 1))
        ic(mg)

    @classmethod
    def ic_vram_info(cls):
        vmi = cls.vram_memoryinfo()
        vmg = SimpleNamespace()
        for name, ctype in vmi._fields_:
            value = getattr(vmi, name)
            setattr(vmg, name, round(value / 1024**3, 1))
        ic(vmg)

    @staticmethod
    def vram_memoryinfo():
        return nvmlDeviceGetMemoryInfo(_gpu)

    def run(self):
        self.pars_args()
        try:
            llm = self.load_model()
            self.chat(llm)
            print('Exiting . . .')
            exit(0)
        except KeyboardInterrupt as e:
            print('\nExiting . . .')
            exit(1)


if __name__ == '__main__':
    Chat().run()

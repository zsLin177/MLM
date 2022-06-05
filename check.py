import torch
import os
from transformers import AutoTokenizer
from transformers import pipeline


def main():
    torch.set_num_threads(1)
    os.environ['CUDA_VISIBLE_DEVICES'] = '4'

    plm = '/data5/slzhou/SRL/SRLasSDGP/SRLasSDGP/bert-base-chinese'
    out_dir = './exp/ner-aishell-mlm'

    tokenizer = AutoTokenizer.from_pretrained(plm)

    fill_mask = pipeline(
        "fill-mask",
        model=out_dir,
        tokenizer=tokenizer
    )
    print(fill_mask('我今天还没有[MASK]饭'))

if __name__ == '__main__':
    main()



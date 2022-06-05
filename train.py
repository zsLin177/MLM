import torch
import os
from transformers import AutoConfig, AutoModel, AutoTokenizer, AutoModelForMaskedLM
from transformers import LineByLineTextDataset
from transformers import DataCollatorForLanguageModeling
from transformers import Trainer, TrainingArguments

def main():
    torch.set_num_threads(1)
    os.environ['CUDA_VISIBLE_DEVICES'] = '4'

    plm = '/data5/slzhou/SRL/SRLasSDGP/SRLasSDGP/bert-base-chinese'
    train_path = './data/aishell-1/combined_train_dev_test.txt'
    out_dir = './exp/aishell-mlm'

    tokenizer = AutoTokenizer.from_pretrained(plm)
    model = AutoModelForMaskedLM.from_pretrained(plm, config=AutoConfig.from_pretrained(plm))

    dataset = LineByLineTextDataset(
        tokenizer=tokenizer,
        file_path=train_path,
        block_size=128,
    )
    data_collator = DataCollatorForLanguageModeling(
        tokenizer=tokenizer, mlm=True, mlm_probability=0.15
    )

    training_args = TrainingArguments(
        output_dir=out_dir,
        overwrite_output_dir=True,
        num_train_epochs=15,
        per_gpu_train_batch_size=16,
        save_steps=10000,
        save_total_limit=5,
        prediction_loss_only=True,
        warmup_ratio=0.1
    )
    trainer = Trainer(
        model=model,
        args=training_args,
        data_collator=data_collator,
        train_dataset=dataset,
    )

    print ('Start a trainer...')
    trainer.train()
    trainer.save_model(out_dir)
    print ('training finished')


if __name__ == '__main__':
    main()



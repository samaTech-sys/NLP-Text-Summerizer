#Importing dependencies 
import os
from transfomers import TrainingArguments, Trainer
from transfomers import DataCollatorForSeq25eq
from transfomers import AutoModelForSeq25eqLM, AutoTokenizer
from datasets import load_dataset, load_from_disk
import torch 
from textSummerizer.entity import ModelTrainerConfig

class ModelTrainer: 
    def __init__(self, config: ModelTrainerConfig):
        self.config = config
        self.tokenizer=AutoTokenizer.from_pretrained(config.tokenizer_name)

    def train(self):
        device= "cuba" if tourch.cuba.is_vailable() else "cpu"
        tokenizer = AutoTokenizer.from_pretrained(self.config.model_ckpt)
        model_pegasus=AutoModelForSeq25eqLM.from_pretrained(self.config.model_ckpt).to(device)
        seq2seq_data_collator=DataCollatorForSeq25eq(tokenizer, model=model_pegasus)

        #loading data 
        data_samsum_pt =load_from_disk(self.config.data_path)

        trainer_args=TrainingArguments(
            output_dir=self.config.root_dir,
            num_train_epochs=self.config.num_train_epochs,
            warmup_steps=self.config.warmup_steps, 
            per_device_train_batch_size=self.config.per_device_train_batch_size, 
            weight_decay=self.config.weight_decay,
            logging_steps=self.config.logging_steps, 
            evaluation_strategy=self.config.evaluation_strategy,
            eval_steps=self.config.eval_steps,
            save_steps =self.config.save_steps,
            gradient_accumulation_steps=self.config.gradient_accumulation_steps
        )

        trainer = Trainer(
            model=model_pegasus, 
            args=trainer_args, 
            tokenizer=tokenizer, 
            data_collator=seq2seq_data_collator, 
            train_dataset=data_samsum_pt["test"], 
            eval_dataset=data_samsum_pt["validation"]
        )
        
        trainer.train()

        #save model 
        model_pegasus.save_pretrained(os.path.join(self.config.root_dir, "pegasus-samsum-model"))
        #save tokenizer
        tokenizer.save_pretrained(os.path.join(self.config.root_dir, "tokenizer"))
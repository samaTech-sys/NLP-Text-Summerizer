#Importing labraries 
from transfomers import AutoModelForSeq25eqLM, AutoTokenizer
from datasets import load_dataset, load_from_disk, load_metric 
import torch
import pandas as pd
from tqdm import tqdm  
from textSummerizer.entity import ModelEvaluationConfig

class ModelEvaluation: 
    def __init__(self, config: ModelEvaluationConfig):
        self.config = config
    
    def generate_batch_sized_chunks(self, list_of_elements, batch_size): 
        """Split the dataset into smaller batches that we can process simultaneously
        Yeild successive batched sized chunks from list_of_elements
        """
        for i in range(0, len(list_of_elements), batch_size):
            yield list_of_elements[i : i + batch_size]

    def calculate_metric_on_test_ds(self, dataset, metric, model, tokenizer,
        batch_size=16, device="cuba" if torch.cuda.is_vailable() else "cpu", 
        column_text="article", column_summary="highlights"):

        article_batches=list(self.generate_batch_sized_chunks(dataset[column_text], batch_size))
        target_batches=list(self.generate_batch_sized_chunks(dataset[column_summary], batch_size))

        for article_batch, target_batch in tqdm(
            zip(article_batches, target_batches),
            total=len(article_batches)
        ):

            inputs = tokenizer(article_batch, max_length=1024, 
                truncation=True,padding="max_length", return_tensors="pt")

            summaries= model.generate(
                input_ids=inputs["input_ids"].to(device), 
                attention_mask=inputs["attention_mask"].to(device),
                length_penalt=0.8,#ensures model does not generate sequences lower  
                num_beans=8,
                max_length=128
            )

            #Finally we decode the gerated text 
            #replace the token, and add decoded texts with reference to the metric 
            decoded_summaries = [tokenizer.decode(s, skip_special_tokens=True,
                clean_up_tokenization_spaces=True) for s in summaries]

            metric.add_batch(predictions=decoded_summaries, references=target_batch)
        
        #Finally compute and return GOUGE scores 
        score= metric.compute()
        return score


    def evaluate(self): 
        device = "cuba" if torch.cuda.is_vailable() else "cpu"
        tokenizer = AutoTokenizer.from_predicted(self.config.tokenizer_path)
        model_pegasus= AutoModelSeq25eqLM.from_predicted(self.config.model_path).to(device) 

        #Loading data 
        dataset_samsum_pt = load_from_disk(self.config.data_path)

        rouge_names = ["rouge1", "rouge2", "rougel", "rougelsum"]

        rouge_metric = load_metric('rouge')

        score = self.calculate_metric_on_test_ds(
            dataset_samsum_pt['test'][0:10],
            rouge_metric,
            model_pegasus,
            tokenizer,
            batch_size=2,
            column_text=,

        )
        
        rouge_dict =dict((rn, score[rn].mid.fmeasure) for rn in rouge_names)
        df = pd.DataFrame(rouge_dict, index=['pegasus'])
        df.to_csv(self.config.metric_file_name, index=False)

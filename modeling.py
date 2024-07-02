import logging

import torch
from torch import nn
from transformers import AutoModelForSequenceClassification, PreTrainedModel, TrainingArguments
from transformers.modeling_outputs import SequenceClassifierOutput

from .arguments import ModelArguments, DataArguments

logger = logging.getLogger(__name__)


class CrossEncoder(nn.Module):
    def __init__(self, hf_model: PreTrainedModel, model_args: ModelArguments, data_args: DataArguments,
                 train_args: TrainingArguments):
        super().__init__()
        self.hf_model = hf_model
        self.model_args = model_args
        self.train_args = train_args
        self.data_args = data_args

        self.config = self.hf_model.config

        self.cross_entropy = nn.CrossEntropyLoss(reduction='mean')
   

    def gradient_checkpointing_enable(self, **kwargs):
        self.hf_model.gradient_checkpointing_enable(**kwargs)

    def forward(self, batch):
        out: SequenceClassifierOutput = self.hf_model(**batch, return_dict=True)
        return out

    @classmethod
    def from_pretrained(
            cls, model_args: ModelArguments, data_args: DataArguments, train_args: TrainingArguments,
            *args, **kwargs
    ):
        hf_model = AutoModelForSequenceClassification.from_pretrained(*args, **kwargs)
        reranker = cls(hf_model, model_args, data_args, train_args)
        return reranker

    def save_pretrained(self, output_dir: str):
        state_dict = self.hf_model.state_dict()
        state_dict = type(state_dict)(
            {k: v.clone().cpu()
             for k,
             v in state_dict.items()})
        self.hf_model.save_pretrained(output_dir, state_dict=state_dict)
        
    def save_checkpoint(self, output_dir: str):
        state_dict = self.hf_model.state_dict()
        state_dict = type(state_dict)(
            {k: v.clone().cpu()
             for k,
             v in state_dict.items()})
        self.hf_model.save_pretrained(output_dir, state_dict=state_dict)
        
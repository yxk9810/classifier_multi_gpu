import logging
import os
from typing import Optional, Dict, List, Union, Tuple, NamedTuple
import time
import math
import torch
import numpy as np
import json
from transformers.trainer import Trainer
from .modeling import CrossEncoder
from torch.utils.data import Dataset
from sklearn.metrics import recall_score, f1_score, precision_score, accuracy_score

logger = logging.getLogger(__name__)


class CETrainer(Trainer):
    def _save(self, output_dir: Optional[str] = None, state_dict=None):
        output_dir = output_dir if output_dir is not None else self.args.output_dir
        os.makedirs(output_dir, exist_ok=True)
        logger.info("Saving model checkpoint to %s", output_dir)
        # Save a trained model and configuration using `save_pretrained()`.
        # They can then be reloaded using `from_pretrained()`
        if not hasattr(self.model, 'save_pretrained'):
            raise NotImplementedError(f'MODEL {self.model.__class__.__name__} ' f'does not support save_pretrained interface')
        else:
            self.model.save_pretrained(output_dir)
        if self.tokenizer is not None and self.is_world_process_zero():
            self.tokenizer.save_pretrained(output_dir)

        # Good practice: save your training arguments together with the trained model
        torch.save(self.args, os.path.join(output_dir, "training_args.bin"))

    def compute_loss(self, model: CrossEncoder, inputs):
        return model(inputs)['loss']

    def evaluate(
        self,
        eval_dataset: Optional[Dataset] = None,
        ignore_keys: Optional[List[str]] = None,
        metric_key_prefix: str = "eval",
    ) -> Dict[str, float]:
        """
        Run evaluation and returns metrics.

        The calling script will be responsible for providing a method to compute metrics, as they are task-dependent
        (pass it to the init `compute_metrics` argument).

        You can also subclass and override this method to inject custom behavior.

        Args:
            eval_dataset (`Dataset`, *optional*):
                Pass a dataset if you wish to override `self.eval_dataset`. If it is a [`~datasets.Dataset`], columns
                not accepted by the `model.forward()` method are automatically removed. It must implement the `__len__`
                method.
            ignore_keys (`List[str]`, *optional*):
                A list of keys in the output of your model (if it is a dictionary) that should be ignored when
                gathering predictions.
            metric_key_prefix (`str`, *optional*, defaults to `"eval"`):
                An optional prefix to be used as the metrics key prefix. For example the metrics "bleu" will be named
                "eval_bleu" if the prefix is "eval" (default)

        Returns:
            A dictionary containing the evaluation loss and the potential metrics computed from the predictions. The
            dictionary also contains the epoch number which comes from the training state.
        """
        # memory metrics - must set up as early as possible
        self._memory_tracker.start()
        eval_dataloader = self.get_eval_dataloader(eval_dataset)
        start_time = time.time()

        eval_loop = self.evaluation_loop
        output = eval_loop(
            eval_dataloader,
            description="Evaluation",
            # No point gathering the predictions if there are no metrics, otherwise we defer to
            # self.args.prediction_loss_only
            prediction_loss_only=True if self.compute_metrics is None else None,
            ignore_keys=ignore_keys,
            metric_key_prefix=metric_key_prefix,
        )
        total_batch_size = self.args.eval_batch_size * self.args.world_size
        if f"{metric_key_prefix}_jit_compilation_time" in output.metrics:
            start_time += output.metrics[f"{metric_key_prefix}_jit_compilation_time"]
        output.metrics.update(
            speed_metrics(
                metric_key_prefix,
                start_time,
                num_samples=output.num_samples,
                num_steps=math.ceil(output.num_samples / total_batch_size),
            )
        )

        self.control = self.callback_handler.on_evaluate(self.args, self.state, self.control, output.metrics)

        self._memory_tracker.stop_and_update_metrics(output.metrics)

        return output.metrics

    def evaluation_loop(
        self,
        dataloader,
        description: str,
        prediction_loss_only: Optional[bool] = None,
        ignore_keys: Optional[List[str]] = None,
        metric_key_prefix: str = "eval",
    ):
        """
        Prediction/evaluation loop, shared by `Trainer.evaluate()` and `Trainer.predict()`.

        Works both with or without labels.
        """
        args = self.args

        model = self._wrap_model(self.model, training=False, dataloader=dataloader)

        batch_size = self.args.eval_batch_size

        logger.info(f"***** Running {description} *****")
        logger.info(f"  Num examples = {self.num_examples(dataloader)}")
        logger.info(f"  Batch size = {batch_size}")
        model.eval()

        self.callback_handler.eval_dataloader = dataloader
        # Do this before wrapping.

        all_loss = []
        all_preds = []
        all_labels = []
        # Main evaluation loop
        for step, inputs in enumerate(dataloader):
            inputs = self._prepare_inputs(inputs)
            out = model(inputs)
            logits = out.logits
            loss = out.loss
            pred = torch.argmax(logits, dim=-1)
            all_preds.extend(pred.tolist())
            all_labels.extend(inputs.labels.tolist())
            all_loss.extend([loss.tolist()]*len(pred))
        loss = sum(all_loss)/len(all_loss)
        # import ipdb;ipdb.set_trace()
        recall = recall_score(all_preds, all_labels, average=None)
        precision = precision_score(all_preds, all_labels, average=None)
        f1 = f1_score(all_preds, all_labels, average=None)
        acc = accuracy_score(all_preds, all_labels)
        macro_recall = recall_score(all_preds, all_labels, average="macro")
        macro_precision = precision_score(all_preds, all_labels, average="macro")
        macro_f1 = f1_score(all_preds, all_labels, average="macro")
        
        weighted_recall = recall_score(all_preds, all_labels, average="weighted")
        weighted_precision = precision_score(all_preds, all_labels, average="weighted")
        weighted_f1 = f1_score(all_preds, all_labels, average="weighted")
        # import ipdb;ipdb.set_trace()
        metrics = {"loss": loss, "precision": precision.tolist(), "recall": recall.tolist(), "f1":f1.tolist(), "macro":[macro_precision, macro_recall, macro_f1], 
                   "weighted":[weighted_precision, weighted_recall, weighted_f1], "acc":acc}
        print(json.dumps(metrics, ensure_ascii=False, indent=2))
        return EvalLoopOutput(predictions=all_preds, label_ids=all_labels, metrics=metrics, num_samples=step)


def speed_metrics(split, start_time, num_samples=None, num_steps=None, num_tokens=None):
    """
    Measure and return speed performance metrics.

    This function requires a time snapshot `start_time` before the operation to be measured starts and this function
    should be run immediately after the operation to be measured has completed.

    Args:
    - split: name to prefix metric (like train, eval, test...)
    - start_time: operation start time
    - num_samples: number of samples processed
    - num_tokens: number of tokens processed
    """
    runtime = time.time() - start_time
    result = {f"{split}_runtime": round(runtime, 4)}
    if runtime == 0:
        return result
    if num_samples is not None:
        samples_per_second = num_samples / runtime
        result[f"{split}_samples_per_second"] = round(samples_per_second, 3)
    if num_steps is not None:
        steps_per_second = num_steps / runtime
        result[f"{split}_steps_per_second"] = round(steps_per_second, 3)
    if num_tokens is not None:
        tokens_per_second = num_tokens / runtime
        result[f"{split}_tokens_per_second"] = round(tokens_per_second, 3)
    return result


class EvalLoopOutput(NamedTuple):
    predictions: Union[np.ndarray, Tuple[np.ndarray]]
    label_ids: Optional[Union[np.ndarray, Tuple[np.ndarray]]]
    metrics: Optional[Dict[str, float]]
    num_samples: Optional[int]

"""
This file is adapted from Huggingface Transformers (
"""

import dataclasses
import json
import logging
import os
from dataclasses import dataclass, field
from typing import Any, Dict, Optional, Tuple

from .adversarial_opt import AdversarialOptBase
from .file_utils import cached_property, is_torch_available, is_torch_tpu_available, torch_required
from .optimizers import OptimizerOptBase, AdamWOpt

if is_torch_available():
    import torch

if is_torch_tpu_available():
    import torch_xla.core.xla_model as xm

logger = logging.getLogger(__name__)


def default_logdir() -> str:
    """
    Same default as PyTorch
    """
    import socket
    from datetime import datetime

    current_time = datetime.now().strftime("%b%d_%H-%M-%S")
    return os.path.join("runs", current_time + "_" + socket.gethostname())


@dataclass
class TrainingArguments:
    output_dir: str = field(
        metadata={"help": "The output directory where the model predictions and checkpoints will be written."}
    )
    overwrite_output_dir: bool = field(
        default=False,
        metadata={
            "help": (
                "Overwrite the content of the output directory."
                "Use this to continue training if output_dir points to a checkpoint directory."
            )
        },
    )

    do_train: bool = field(default=True, metadata={"help": "Whether to run training."})
    do_eval: bool = field(default=True, metadata={"help": "Whether to run eval on the dev set."})
    do_predict: bool = field(default=False, metadata={"help": "Whether to run predictions on the test set."})
    evaluate_during_training: bool = field(
        default=True, metadata={"help": "Run evaluation during training at each logging step."},
    )

    per_device_train_batch_size: int = field(
        default=8, metadata={"help": "Batch size per GPU/TPU core/CPU for training."}
    )
    per_device_eval_batch_size: int = field(
        default=8, metadata={"help": "Batch size per GPU/TPU core/CPU for evaluation."}
    )

    per_gpu_train_batch_size: Optional[int] = field(
        default=None,
        metadata={
            "help": "Deprecated, the use of `--per_device_train_batch_size` is preferred. "
                    "Batch size per GPU/TPU core/CPU for training."
        },
    )
    per_gpu_eval_batch_size: Optional[int] = field(
        default=None,
        metadata={
            "help": "Deprecated, the use of `--per_device_eval_batch_size` is preferred."
                    "Batch size per GPU/TPU core/CPU for evaluation."
        },
    )

    gradient_accumulation_steps: int = field(
        default=1,
        metadata={"help": "Number of updates steps to accumulate before performing a backward/update pass."},
    )
    optimizer_opt: OptimizerOptBase = field(
        default=AdamWOpt(),
        metadata={'help':
                      """
                      The configurations of optimizers. Must be a subclass of OptimizerOptBase.
                      Default value is AdamWOpt()
                      """}
    )
    max_grad_norm: float = field(default=1.0, metadata={"help": "Max gradient norm."})

    num_train_epochs: float = field(default=3.0, metadata={"help": "Total number of training epochs to perform."})
    max_steps: int = field(
        default=-1,
        metadata={"help": "If > 0: set total number of training steps to perform. Override num_train_epochs."},
    )
    warmup_steps: int = field(default=0, metadata={"help": "Linear warmup over warmup_steps."})

    logging_dir: Optional[str] = field(default_factory=default_logdir, metadata={"help": "Tensorboard log dir."})
    logging_first_step: bool = field(default=False, metadata={"help": "Log and eval the first global_step"})
    logging_steps: int = field(default=500, metadata={"help": "Log every X updates steps."})
    save_steps: int = field(default=500, metadata={"help": "Save checkpoint every X updates steps."})
    save_total_limit: Optional[int] = field(
        default=None,
        metadata={
            "help": (
                "Limit the total amount of checkpoints."
                "Deletes the older checkpoints in the output_dir. Default is unlimited checkpoints"
            )
        },
    )
    no_cuda: bool = field(default=False, metadata={"help": "Do not use CUDA even when it is available"})
    seed: int = field(default=42, metadata={"help": "random seed for initialization"})

    fp16: bool = field(
        default=False,
        metadata={"help": "Whether to use 16-bit (mixed) precision (through NVIDIA apex) instead of 32-bit"},
    )
    fp16_opt_level: str = field(
        default="O1",
        metadata={
            "help": (
                "For fp16: Apex AMP optimization level selected in ['O0', 'O1', 'O2', and 'O3']."
                "See details at https://nvidia.github.io/apex/amp.html"
            )
        },
    )
    local_rank: int = field(default=-1, metadata={"help": "For distributed training: local_rank"})

    tpu_num_cores: Optional[int] = field(
        default=None, metadata={"help": "TPU: Number of TPU cores (automatically passed by launcher script)"}
    )
    tpu_metrics_debug: bool = field(
        default=False,
        metadata={"help": "Deprecated, the use of `--debug` is preferred. TPU: Whether to print debug metrics"},
    )
    debug: bool = field(default=False, metadata={"help": "Whether to print debug metrics on TPU"})

    dataloader_drop_last: bool = field(
        default=False, metadata={"help": "Drop the last incomplete batch if it is not divisible by the batch size."}
    )
    eval_steps: int = field(default=1000, metadata={"help": "Run an evaluation every X steps."})

    past_index: int = field(
        default=-1,
        metadata={"help": "If >=0, uses the corresponding part of the output as the past state for next step."},
    )

    adv_opt: AdversarialOptBase = field(
        default=None,
        metadata={'help':
                      "The options that we use to run adversarial training based on gradient."
                      "If None, no adversarial training is performed. "
                      "Default is None."}
    )
    @property
    def train_batch_size(self) -> int:
        """
        The actual batch size for training (may differ from :obj:`per_gpu_train_batch_size` in distributed training).
        """
        if self.per_gpu_train_batch_size:
            logger.warning(
                "Using deprecated `--per_gpu_train_batch_size` argument which will be removed in a future "
                "version. Using `--per_device_train_batch_size` is preferred."
            )
        per_device_batch_size = self.per_gpu_train_batch_size or self.per_device_train_batch_size
        return per_device_batch_size * max(1, self.n_gpu)

    @property
    def eval_batch_size(self) -> int:
        """
        The actual batch size for evaluation (may differ from :obj:`per_gpu_eval_batch_size` in distributed training).
        """
        if self.per_gpu_eval_batch_size:
            logger.warning(
                "Using deprecated `--per_gpu_eval_batch_size` argument which will be removed in a future "
                "version. Using `--per_device_eval_batch_size` is preferred."
            )
        per_device_batch_size = self.per_gpu_eval_batch_size or self.per_device_eval_batch_size
        return per_device_batch_size * max(1, self.n_gpu)

    @cached_property
    @torch_required
    def _setup_devices(self) -> Tuple["torch.device", int]:
        logger.info("PyTorch: setting up devices")
        if self.no_cuda:
            device = torch.device("cpu")
            n_gpu = 0
        elif is_torch_tpu_available():
            device = xm.xla_device()
            n_gpu = 0
        elif self.local_rank == -1:
            # if n_gpu is > 1 we'll use nn.DataParallel.
            # If you only want to use a specific subset of GPUs use `CUDA_VISIBLE_DEVICES=0`
            # Explicitly set CUDA to the first (index 0) CUDA device, otherwise `set_device` will
            # trigger an error that a device index is missing. Index 0 takes into account the
            # GPUs available in the environment, so `CUDA_VISIBLE_DEVICES=1,2` with `cuda:0`
            # will use the first GPU in that env, i.e. GPU#1
            device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
            n_gpu = torch.cuda.device_count()
        else:
            # Here, we'll use torch.distributed.
            # Initializes the distributed backend which will take care of sychronizing nodes/GPUs
            torch.distributed.init_process_group(backend="nccl")
            device = torch.device("cuda", self.local_rank)
            n_gpu = 1

        if device.type == "cuda":
            torch.cuda.set_device(device)

        return device, n_gpu

    @property
    @torch_required
    def device(self) -> "torch.device":
        """
        The device used by this process.
        """
        return self._setup_devices[0]

    @property
    @torch_required
    def n_gpu(self):
        """
        The number of GPUs used by this process.
        Note:
            This will only be greater than one when you have multiple GPUs available but are not using distributed
            training. For distributed training, it will always be 1.
        """
        return self._setup_devices[1]

    def to_json_string(self):
        """
        Serializes this instance to a JSON string.
        """
        return json.dumps(dataclasses.asdict(self), indent=2)

    def to_sanitized_dict(self) -> Dict[str, Any]:
        """
        Sanitized serialization to use with TensorBoard’s hparams
        """
        d = dataclasses.asdict(self)
        valid_types = [bool, int, float, str]
        if is_torch_available():
            valid_types.append(torch.Tensor)
        return {k: v if type(v) in valid_types else str(v) for k, v in d.items()}

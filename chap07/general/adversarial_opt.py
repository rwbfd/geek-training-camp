# coding = 'utf-8'

import logging
from dataclasses import dataclass, field

"""
These are options for adversarial attacks based on gradient methods. 

The implemented methods include:

1. FGSM (Fast Gradient Sign Method, https://arxiv.org/abs/1412.6572).
2. FGM (Fast Gradient Method).
3. FreeLB (Free Large-Batch).
4. SMART (Smoothness-inducing Adversarial Regularization).
"""


@dataclass
class AdversarialOptBase:
    pass


@dataclass
class FGSMOpt(AdversarialOptBase):
    eps: float = field(
        default=1e-5,
        metadata={
            'help': "The noise coefficient to multiply the sign of gradient."
                    "Controls the extent of noise."}
    )


@dataclass
class FGMOpt(AdversarialOptBase):
    eps: float = field(
        default=1e-5,
        metadata={'help':
                      "The noise coefficient to multiply the sign of gradient divided by its norm."
                      "Controls the extent of noise."}
    )


@dataclass
class FreeLBOpt(AdversarialOptBase):
    adv_init_msg: float = field(
        default=0,
        metadata={'help':
                      """
                      TO DO.
            
                      """}

    )

    norm_type: str = field(
        default='l2',
        metadata={'help':
                      """
                      The norm to use. 
                      Must be either l2 or linf.
                      Default is l2.
                      """}
    )

    adv_steps: int = field(
        default=5,
        metadata={'help':
                      """
                      The number of adversarial training steps. 
                      Default is 5.
                      """}
    )

    adv_lr: int = field(
        default=1e-5,
        metadata={'help':
                      """
                      The learning rate for the adversarial training steps.
                      Default value is 1e-5
                      """}
    )

    def __post_init__(self):
        if self.norm_type not in {'l2', 'linf'}:
            logging.error("The norm type must be either l2 or linf")

        if self.adv_steps <= 0:
            logging.error("The adversarial training steps must be an integer that is larger than 0")

        if self.adv_lr <= 0:
            logging.error("The adversarial learning rate must be larger than 0.")

# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
# http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.

"""
Financial Foundation Model Pre-training — Decoder-Only (NeMo AutoModel)

Pretrains a decoder causal language model on financial transaction
sequences using NeMo AutoModel's TrainFinetuneRecipeForNextTokenPrediction.

Custom components:
  - FinancialTabularTokenizer: Domain-specific tokenizer (12 tokens per txn)
  - FinancialCLMDataset: Next-token prediction data loading (no MLM masking)

Launch methods
--------------
Multi-GPU (recommended):
    torchrun --nproc-per-node=8 scripts/train_decoder_model.py \
        -c configs/pretrain_financial_decoder.yaml \
        --dataset.data_path data/decoder_corpus/train_corpus.txt \
        --validation_dataset.data_path data/decoder_corpus/val_corpus.txt

Single GPU (testing):
    python scripts/train_decoder_model.py \
        -c configs/pretrain_financial_decoder.yaml \
        --dataset.data_path data/decoder_corpus/train_corpus.txt
"""

import sys
from pathlib import Path

BLUEPRINT_ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(BLUEPRINT_ROOT))


def main():
    from nemo_automodel.components.config._arg_parser import parse_args_and_load_config
    from nemo_automodel.recipes.llm.train_ft import TrainFinetuneRecipeForNextTokenPrediction

    cfg = parse_args_and_load_config()

    model_cfg = cfg.model.config
    target = getattr(model_cfg, '_target_', None)
    if target is not None:
        arch = target.__name__ if isinstance(target, type) else str(target).rsplit('.', 1)[-1]
    else:
        arch = "unknown"

    print("\n" + "=" * 60)
    print("Financial Foundation Model — Decoder-Only Pretraining")
    print("=" * 60)
    print(f"  Architecture: {arch}")
    for key in ['hidden_size', 'n_embd']:
        try:
            print(f"  Hidden size:  {getattr(model_cfg, key)}")
            break
        except (AttributeError, KeyError):
            continue
    for key in ['num_hidden_layers', 'n_layer']:
        try:
            print(f"  Layers:       {getattr(model_cfg, key)}")
            break
        except (AttributeError, KeyError):
            continue
    print(f"  Vocab size:   {model_cfg.vocab_size}")
    print(f"  Max steps:    {cfg.step_scheduler.max_steps}")
    print(f"  Batch size:   {cfg.step_scheduler.global_batch_size} global, "
          f"{cfg.step_scheduler.local_batch_size} local")
    print("=" * 60 + "\n")

    # HACK: Fix FSDP2 meta device initialization bug for HuggingFace models
    # FSDP2 materializes from `meta` device but HF models lack `reset_parameters()`.
    # We monkey-patch FSDP2Manager.parallelize to materialize and initialize on CUDA first.
    import torch
    from nemo_automodel.components.distributed.fsdp2 import FSDP2Manager
    original_parallelize = FSDP2Manager.parallelize
    
    def patched_parallelize(self, model):
        # Move off meta device
        device = torch.device("cuda", torch.cuda.current_device())
        model.to_empty(device=device)
        
        @torch.no_grad()
        def _init_weights(module):
            if isinstance(module, torch.nn.Linear):
                torch.nn.init.normal_(module.weight, mean=0.0, std=0.02)
                if getattr(module, "bias", None) is not None:
                    torch.nn.init.zeros_(module.bias)
            elif isinstance(module, torch.nn.Embedding):
                torch.nn.init.normal_(module.weight, mean=0.0, std=0.02)
                if getattr(module, "padding_idx", None) is not None:
                    module.weight[module.padding_idx].zero_()
            elif "Norm" in module.__class__.__name__:
                if hasattr(module, "weight") and module.weight is not None:
                    torch.nn.init.ones_(module.weight)
                if getattr(module, "bias", None) is not None:
                    torch.nn.init.zeros_(module.bias)
                    
        model.apply(_init_weights)
        return original_parallelize(self, model)
        
    FSDP2Manager.parallelize = patched_parallelize

    recipe = TrainFinetuneRecipeForNextTokenPrediction(cfg)
    recipe.setup()
    recipe.run_train_validation_loop()


if __name__ == "__main__":
    main()

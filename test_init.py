import sys
sys.path.append('/export/home/sobhan/cs25mtech14005/transaction-foundation-model')
from nemo_automodel.components.config._arg_parser import parse_args_and_load_config
from nemo_automodel.recipes.llm.train_ft import TrainFinetuneRecipeForNextTokenPrediction
import torch

# Load config
import omegaconf
cfg = omegaconf.OmegaConf.load("configs/pretrain_financial_decoder.yaml")
cfg.clip_grad_norm = omegaconf.OmegaConf.create({"max_norm": 1.0})

recipe = TrainFinetuneRecipeForNextTokenPrediction(cfg)
recipe.setup()

model = recipe.model_parts[0]
print("Before init:", model.model.embed_tokens.weight.abs().mean().item())

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
    elif isinstance(module, torch.nn.LayerNorm) or "RMSNorm" in module.__class__.__name__:
        if hasattr(module, "weight") and module.weight is not None:
            torch.nn.init.ones_(module.weight)
        if getattr(module, "bias", None) is not None:
            torch.nn.init.zeros_(module.bias)

model.apply(_init_weights)
print("After init:", model.model.embed_tokens.weight.abs().mean().item())

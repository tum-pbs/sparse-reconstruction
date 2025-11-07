from GIFt.strategies.lora import LoRAConfigMixin, LoRALinearFineTuningStrategy, LoRAConvFineTuningStrategy
from GIFt.strategies import FineTuningStrategy, merger_strategy
from GIFt.modules.lora import LoRAConv2d,LoRALinear
import GIFt.utils.factories as fts

from core.image_models.models import CustomDiTTransformer2DModel
from core.training.transformer_image_flow import TransformerImageFlow
class LoRATransformerStrategy(LoRAConfigMixin,FineTuningStrategy):
    
    def __init__(self, 
                 rank: int = 3, 
                 lora_alpha: float | None = None, 
                 lora_dropout: float = 0, 
                 train_bias: bool = False,) -> None:
        LoRAConfigMixin.__init__(self,rank,lora_alpha,lora_dropout,train_bias)
        FineTuningStrategy.__init__(self)
        self.regisier_constraint_types([CustomDiTTransformer2DModel,TransformerImageFlow])
        self.register_cap(
            check_func=fts.mc_name_equal2_sequence(["to_q","to_k","to_v"]),
            act_func=fts.ma_replace(LoRALinear),
            act_para=self.lora_configs()
        )
from GIFt.strategies.lora import LoRAConfigMixin
from GIFt.strategies import FineTuningStrategy
from GIFt.modules.lora import LoRAConv2d
import GIFt.utils.factories as fts


from core.image_models.models import UNet
from core.training.unet_image_flow import UNetImageFlow
class LoRAUNetFineTuningStrategy(LoRAConfigMixin,FineTuningStrategy):
    
    def __init__(self, 
                 rank: int = 3, 
                 lora_alpha: float | None = None, 
                 lora_dropout: float = 0, 
                 train_bias: bool = False,) -> None:
        LoRAConfigMixin.__init__(self,rank,lora_alpha,lora_dropout,train_bias)
        FineTuningStrategy.__init__(self)
        self.regisier_constraint_types([UNet,UNetImageFlow])
        self.register_cap(
            check_func=fts.mc_name_equal2("project_in"),
            act_func=fts.ma_replace(LoRAConv2d),
            act_para=self.lora_configs()
        )
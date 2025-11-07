from GIFt.strategies import FineTuningStrategy,merger_strategy
from GIFt.strategies.svdiff import SVDiffLinearFineTuningStrategy, SVDiffConvFineTuningStrategy
from GIFt.utils import factories as fts
from GIFt.modules.svdiff import SVDiffLinear

from core.image_models.models import CustomDiTTransformer2DModel
from core.training.transformer_image_flow import TransformerImageFlow

class SVDiffAttentionStrategy(FineTuningStrategy):
    
    def __init__(self,train_bias=False) -> None:
        super().__init__()
        self.regisier_constraint_types([CustomDiTTransformer2DModel,TransformerImageFlow])
        self.register_cap(
            check_func=fts.mc_name_equal2_sequence(["to_q","to_k","to_v"]),
            act_func=fts.ma_replace(SVDiffLinear),
            act_para={"train_bias":train_bias}
        )
        
class PoseidonSVDiffAttentionStrategy(FineTuningStrategy):
    
    def __init__(self,train_bias: bool = False) -> None:
        FineTuningStrategy.__init__(self,
                                    [
                                        (fts.mc_name_equal2_sequence(["query","key","value"]),
                                         fts.ma_replace(SVDiffLinear),
                                         {"train_bias":train_bias}),
                                    ],
                                    )

def PoseidonSVDiffAllStrategy(
    train_bias: bool = False) -> FineTuningStrategy:
    return merger_strategy([SVDiffLinearFineTuningStrategy(train_bias),
                            SVDiffConvFineTuningStrategy(train_bias)],)

def PoseidonSVDiffFullStrategy(
    train_bias: bool = True) -> FineTuningStrategy:
    return merger_strategy([SVDiffLinearFineTuningStrategy(train_bias),
                            SVDiffConvFineTuningStrategy(train_bias)],
                            additional_para_caps=
                            [
                                (
                                lambda *args,**kwargs: True,
                                lambda *args,**kwargs: None,
                                {} 
                                )
                            ])
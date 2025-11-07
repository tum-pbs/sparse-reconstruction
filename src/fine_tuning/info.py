from dataclasses import dataclass
from typing import Sequence,Optional
from src.utils import table_info
from GIFt.strategies import FineTuningStrategy
from GIFt.utils import get_class_name
from lightning import LightningModule

@dataclass        
class FineTuningInfo:
    path_pretrained_weights:Optional[str]=None
    name_fine_tuning_strategy:Optional[str]=None
    num_para_before:Optional[str]=None
    num_para_after:Optional[str]=None
    info_parameters:Optional[Sequence[Sequence[str]]]=None
        
def fill_finetuning_info(module:LightningModule,
                          strategy:FineTuningStrategy,
                          info:FineTuningInfo)->str:
    index=0
    num_para_after=0
    fine_tuning_parameters=[]
    for name,p in module.named_parameters():
        if p.requires_grad:
            p_num=p.numel()
            num_para_after+=p_num
            fine_tuning_parameters.append([str(index),name,str(list(p.shape)),str(p_num)])
            index+=1
    info.num_para_after=num_para_after
    info.info_parameters=fine_tuning_parameters
    info.name_fine_tuning_strategy=get_class_name(strategy)
    return info

def plain_fine_tuning_info(info:FineTuningInfo)->str:
    table_msg,h_line=table_info(info.info_parameters,
                         ["Index","Name","Shape","Number of elements"],
                         return_hline=True)
    msg=[h_line,h_line]
    msg.append(f"Fine-tuning enabled with stragegy {info.name_fine_tuning_strategy}")
    msg.append(f"Path to the pretrained weights: {info.path_pretrained_weights}")
    msg.append(f"Number of parameters before fine-tuning: {info.num_para_before}")
    msg.append(f"Number of parameters after fine-tuning: {info.num_para_after} ({info.num_para_after/info.num_para_before:.2%})")
    msg.append("Fine-tuning parameters:")
    msg.append(table_msg)
    msg.append(h_line)
    return "\n".join(msg)

def wandb_text_fine_tuning_info(info:FineTuningInfo)->Sequence[dict]:
    return [
        {
            "key":"Fine-tuning info",
            "columns":["strategy","path_pretrained_weights","num_para_before","num_para_after"],
            "data":[[info.name_fine_tuning_strategy,
                    info.path_pretrained_weights,
                    info.num_para_before,
                    "{}({:.2%})".format(info.num_para_after,
                                    info.num_para_after/info.num_para_before),
                    ]] 
        },
        {
            "key":"Fine-tuning parameters",
            "columns":["Index","Name","Shape","Number of elements"],
            "data":info.info_parameters,    
        }
    ]


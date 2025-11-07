from conflictfree import *
from conflictfree.utils import *
from conflictfree.length_model import *

class UniProjectionLength(LengthModel):
    """
    Rescale the length of the target vector based on the projection of the gradients on the target vector.
    """

    def __init__(self):
        super().__init__()
        

    def rescale_length(self, 
                       target_vector:torch.Tensor,
                       gradients:Optional[torch.Tensor]=None,
                       losses:Optional[Sequence]=None)->torch.Tensor:

        unit_target_vector = unit_vector(target_vector)
        length = self.get_length(
            target_vector=target_vector,
            unit_target_vector=unit_target_vector,
            gradients=gradients,
            losses=losses
        ) 

        return length * unit_target_vector


    def get_length(self, target_vector:Optional[torch.Tensor]=None,
                         unit_target_vector:Optional[torch.Tensor]=None,
                         gradients:Optional[torch.Tensor]=None,
                         losses:Optional[Sequence]=None)->torch.Tensor:
        

        if gradients is None:
            raise ValueError("The ProjectLength model requires gradients information.")
        if unit_target_vector is None:
            unit_target_vector = unit_vector(target_vector)
        return torch.sum(torch.stack([torch.dot(unit_vector(grad_i),unit_target_vector) for grad_i in gradients]))
    
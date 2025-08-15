from dataclasses import dataclass
import torch
from typing import List

@dataclass
class CineModalInput:
    temporal_img: torch.Tensor
    spatial_img: torch.Tensor
    img_2ch: torch.Tensor
    img_3ch: torch.Tensor
    img_4ch: torch.Tensor
    temporal_optical_img: torch.Tensor
    spatial_optical_img: torch.Tensor
    CH_temporal_img_optical_ch2: torch.Tensor
    CH_temporal_img_optical_ch3: torch.Tensor
    CH_temporal_img_optical_ch4: torch.Tensor
    text: str
    prompt_text: str
    clinical_indicators: torch.Tensor
    labels_c: torch.Tensor
    labels_Y: torch.Tensor
    labels_time: torch.Tensor
    case_name: str

    def to(self, device):
        for field in self.__dataclass_fields__:
            val = getattr(self, field)
            if isinstance(val, torch.Tensor):
                setattr(self, field, val.to(device))
            elif isinstance(val, list) and isinstance(val[0], torch.Tensor):
                setattr(self, field, [v.to(device) for v in val])
            else:
                pass
        return self

def cine_collate_fn(batch: List[CineModalInput]):
    collated = {}
    for field in CineModalInput.__dataclass_fields__:
        values = [getattr(sample, field) for sample in batch]

        if isinstance(values[0], torch.Tensor):
            collated[field] = torch.stack(values)
        else:
            collated[field] = values
    return CineModalInput(**collated)

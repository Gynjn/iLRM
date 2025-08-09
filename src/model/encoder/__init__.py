from typing import Optional

from .encoder import Encoder
from .encoder_scale import EncoderScale, EncoderScaleCfg

ENCODERS = {
    "scale": EncoderScale,
}

EncoderCfg = EncoderScaleCfg


def get_encoder(cfg: EncoderCfg) -> Encoder:
    encoder = ENCODERS[cfg.name]
    encoder = encoder(cfg)
    return encoder

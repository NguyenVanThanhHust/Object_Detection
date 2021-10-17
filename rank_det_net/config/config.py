from detectron2.config import CfgNode as CN

def add_rankdet_net_config(cfg):
    _C = cfg
    _C.MODEL.RANK_ROI_HEAD = CN()
    _C.MODEL.RANK_ROI_HEAD.NAME = "RankROIHeads"

    return _C
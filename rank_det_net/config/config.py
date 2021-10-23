from detectron2.config import CfgNode as CN

def add_rankdet_net_config(cfg):
    _C = cfg
    _C.MODEL.RANK_ROI_HEAD = CN()
    _C.MODEL.RANK_ROI_HEAD.NAME = "RankROIHeads"
    _C.SOLVER.OHEM_USE_NMS = True
    _C.SOLVER.OHEM_NMS_THRESH = 0.1
    return _C
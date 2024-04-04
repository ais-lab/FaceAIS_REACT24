import traceback
from dataclasses import dataclass

import torch

from metrics.FRC import compute_FRC, compute_FRC_mp
from metrics.FRD import compute_FRD, compute_FRD_mp
from metrics.FRDvs import compute_FRDvs
from metrics.FRRea import compute_fid
from metrics.FRVar import compute_FRVar
from metrics.S_MSE import compute_s_mse
from metrics.TLCC import compute_TLCC, compute_TLCC_mp


class AverageMeter(object):
    """Computes and stores the average and current value"""

    def __init__(self):
        self.reset()

    def reset(self):
        self.val = 0
        self.avg = 0
        self.sum = 0
        self.count = 0

    def update(self, val, n=1):
        self.val = val
        self.sum += val * n
        self.count += n
        self.avg = self.sum / self.count


@dataclass
class ComputeReturn:
    TLCC: AverageMeter
    FRC: AverageMeter
    FRD: AverageMeter
    FRDvs: AverageMeter
    FRVar: AverageMeter
    smse: AverageMeter
    FRRea: AverageMeter


def compute(
    dataset_path: str,
    listener_pred: torch.Tensor,
    speaker_gt: torch.Tensor,
    listener_gt: torch.Tensor,
    val_test: str,
    past_metrics: ComputeReturn = None,
    fid_dir: str = None,
    list_fake: list = None,
    list_real: list = None,
    device: str = "cpu",
    p=8,
) -> ComputeReturn:
    # If you have problems running function compute_TLCC_mp, please replace this function with function compute_TLCC
    TLCC = compute_TLCC_mp(listener_pred, speaker_gt, p=p)
    print("TLCC: ", TLCC)
    # If you have problems running function compute_FRC_mp, please replace this function with function compute_FRC
    FRC = compute_FRC_mp(
        dataset_path,
        listener_pred,
        listener_gt,
        val_test=val_test,
        p=p,
    )
    print("FRC: ", FRC)
    # If you have problems running function compute_FRD_mp, please replace this function with function compute_FRD
    FRD = compute_FRD_mp(
        dataset_path,
        listener_pred,
        listener_gt,
        val_test=val_test,
        p=p,
    )
    print("FRD: ", FRD)
    FRDvs = compute_FRDvs(listener_pred)
    print("FRDvs: ", FRDvs.item())
    FRVar = compute_FRVar(listener_pred)
    print("FRVar: ", FRVar.item())
    smse = compute_s_mse(listener_pred)
    print("smse: ", smse.item())
    try:
        if fid_dir:
            FRRea = compute_fid(fid_dir, device=device)
        elif list_fake and list_real:
            FRRea = compute_fid(
                real_path=list_real,
                fake_path=list_fake,
                device=device,
            )
        else:
            FRRea = -1
    except Exception as e:
        traceback.print_exc()
        FRRea = -1
    print("FRRea: ", FRRea.item())

    if past_metrics is None:
        past_metrics = ComputeReturn(
            TLCC=AverageMeter(),
            FRC=AverageMeter(),
            FRD=AverageMeter(),
            FRDvs=AverageMeter(),
            FRVar=AverageMeter(),
            smse=AverageMeter(),
            FRRea=AverageMeter(),
        )

    past_metrics.TLCC.update(TLCC)
    past_metrics.FRC.update(FRC)
    past_metrics.FRD.update(FRD)
    past_metrics.FRDvs.update(FRDvs)
    past_metrics.FRVar.update(FRVar)
    past_metrics.smse.update(smse)
    past_metrics.FRRea.update(FRRea)

    return past_metrics

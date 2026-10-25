"""Check the actual GPU environment before downloading/loading weights."""
import platform
import argparse
import os
import subprocess
import importlib.metadata
from pathlib import Path
import torch
from transformers.models.qwen3_5.modeling_qwen3_5 import is_fast_path_available
from src.utils import config, save

def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--config', default='config.yaml')
    args = parser.parse_args()
    cfg=config(args.config); assert torch.cuda.is_available(), 'CUDA GPU unavailable; check NVIDIA driver and CUDA PyTorch wheel'
    assert torch.cuda.is_bf16_supported(), 'BF16 unsupported on this GPU'
    prop=torch.cuda.get_device_properties(0)
    report=dict(deltanet_fast_path_available=is_fast_path_available,gpu=prop.name,vram_gib=prop.total_memory/2**30,torch=torch.__version__,
        cuda_runtime=torch.version.cuda,python=platform.python_version(),platform=platform.platform(),
        packages={k:importlib.metadata.version(k) for k in ['transformers','torchvision','accelerate','datasets']},
        nvidia_smi=subprocess.check_output(['nvidia-smi'],text=True),
        visible_gpu_count=torch.cuda.device_count(),
        slurm_job_id=os.environ.get('SLURM_JOB_ID'),
        cuda_visible_devices=os.environ.get('CUDA_VISIBLE_DEVICES'),
        gpus=[dict(name=torch.cuda.get_device_properties(i).name,
                   vram_gib=torch.cuda.get_device_properties(i).total_memory/2**30)
              for i in range(torch.cuda.device_count())])
    save(Path(cfg['output'])/'hardware.json',report)
    if any(g['vram_gib'] < 39 for g in report['gpus']):
        raise SystemExit('Request GPUs with nominal 40 GB or more each (A100/H100/L40S). Two smaller GPUs do not pool memory in this implementation.')
    print(f"GPU ready: {prop.name}, {report['vram_gib']:.1f} GiB; saved hardware.json")
if __name__=='__main__': main()

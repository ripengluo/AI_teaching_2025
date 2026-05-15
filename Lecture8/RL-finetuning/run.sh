#!/usr/bin/env bash
python -u ./rl_smile_adapter.py --base_ckpt ./models/celebA_sf2m/sf2m_celeba_64_cond_Sex.pth --out_dir ./outputs/smile_adapter_positive --oracle_mix_ratio 0.5 "$@"

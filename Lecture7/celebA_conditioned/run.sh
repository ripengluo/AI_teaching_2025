#python sf2m_celebA.py --train --total_steps 300000 --cond_attrs Sex --x_dim 64 --batch_size 64 
#python sf2m_celebA.py --sample_sde --sex 0 --n_samples 8 
#
#python sf2m_celebA.py --train --total_steps 300000 --cond_attrs Sex Eyeglasses Young --x_dim 64 --batch_size 64
python sf2m_celebA.py --sample_sde --ckpt models/celebA_sf2m/sf2m_celeba_64_cond_Sex-Eyeglasses-Young.pth --cond_attrs Sex Eyeglasses Young --cond_bits 0 1 0

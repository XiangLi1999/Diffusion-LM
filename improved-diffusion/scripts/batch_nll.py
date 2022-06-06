import os, sys, glob
full_lst = glob.glob(sys.argv[1])
pattern_ = 'model' if len(sys.argv) < 2 else sys.argv[2]
clamp = 'clamp' if len(sys.argv) <= 3 else sys.argv[3]
print(f'pattern_ = {pattern_}', sys.argv[2])

for lst in full_lst:
    print(lst)
    try:
        tgt = sorted(glob.glob(f"{lst}/{pattern_}*pt"))[-1]
        lst = os.path.split(lst)[1]
        print(lst)
        num = 1
    except:
        continue

    COMMAND = f'python scripts/nll.py --clip_denoised False ' \
        f'--model_path {tgt} ' \
        f'--out_dir diffusion_lm/improved_diffusion/scores_eval2_valid_None ' \
              f'--num_samples 64 --split valid --clamp {clamp}'
    print(COMMAND)
    os.system(COMMAND)

    COMMAND = f'python scripts/nll.py --clip_denoised False ' \
              f'--model_path {tgt} ' \
              f'--out_dir diffusion_lm/improved_diffusion/scores_eval2_valid_None ' \
              f'--num_samples 64 --split train --clamp {clamp}'
    print(COMMAND)
    os.system(COMMAND)
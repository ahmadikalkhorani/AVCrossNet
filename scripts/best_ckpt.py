

from ast import parse
import glob 
import argparse

def extract_metrc(x):
    if "_neg_si_sdi_" in x:
        k = "_neg_si_sdi_"
    elif "_neg_si_sdr_" in x:
        k = "_neg_si_sdr_"
    return float(x.split(k)[-1].replace(".ckpt", ""))

if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('--path', type=str, default='checkpoints/*') 
    
    args = parser.parse_args()
    
    
    folder = args.path 
    files = [f for f in glob.glob(f"{folder}/**/*.ckpt", recursive=True) if "epoch" in f and "last" not in f]
    files = list(set(files))
    
    for k in ["_neg_si_sdi_", "_neg_si_sdr_"]:
        if k in files[-1]:
            files.sort(key=lambda x: extract_metrc(x), reverse=False)
    print(files[0])
        
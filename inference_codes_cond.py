import os
import glob
import tqdm
import torch
import argparse
from scipy.io.wavfile import write
from omegaconf import OmegaConf

from model.generator import Generator

from utils.utils import read_wav_np
from utils.stft import TacotronSTFT


def main(args):
    checkpoint = torch.load(args.checkpoint_path)
    if args.config is not None:
        hp = OmegaConf.load(args.config)
    else:
        hp = OmegaConf.create(checkpoint['hp_str'])

    model = Generator(hp).cuda()
    saved_state_dict = checkpoint['model_g']
    new_state_dict = {}
    
    for k, v in saved_state_dict.items():
        try:
            new_state_dict[k] = saved_state_dict['module.' + k]
        except:
            new_state_dict[k] = v
    model.load_state_dict(new_state_dict)
    model.eval(inference=True)

    if args.cond_file:
        speaker_cond = torch.load(args.cond_file, map_location='cpu')
        
    os.makedirs(args.output_folder, exist_ok=True)
    with torch.no_grad():
        for codepath in tqdm.tqdm(glob.glob(os.path.join(args.input_folder, '*.pth'))):

            code = torch.load(codepath, map_location='cpu')

            if len(code.shape) == 1:
                code = code.unsqueeze(0)
            code = code.cuda()

            if not args.cond_file:
                speaker_cond_path = os.path.join(args.cond_folder, os.path.basename(codepath).replace('.pth', '_speaker_cond.pth'))
                speaker_cond = torch.load(speaker_cond_path, map_location='cpu').float()
                speaker_cond = speaker_cond.cuda()

            audio = model.inference(code, speaker_cond)
            audio = audio.cpu().detach().numpy()

            basename = os.path.basename(codepath)
            out_path = os.path.join(args.output_folder, basename.replace('.pth','.wav'))
            write(out_path, hp.audio.sampling_rate, audio)


if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('-c', '--config', type=str, default=None,
                        help="yaml file for config. will use hp_str from checkpoint if not given.")
    parser.add_argument('-p', '--checkpoint_path', type=str, required=True,
                        help="path of checkpoint pt file for evaluation")
    parser.add_argument('-i', '--input_folder', type=str, required=True,
                        help="directory of mel-spectrograms to invert into raw audio.")
    parser.add_argument('-o', '--output_folder', type=str, default=None,
                        help="directory which generated raw audio is saved.")
    parser.add_argument('-s', '--cond_file', default=None,
                        help='conditioning file to be used.')
    parser.add_argument('-f', '--cond_folder', default=None,
                        help='folder containing conditioning files to be used.')
    args = parser.parse_args()
#    assert args.cond_file is not None and args.cond_folder is not None, 'You must provide either conditioning file or folder.'

    main(args)

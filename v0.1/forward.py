import pprint

import torch
from loguru import logger
from pathlib import Path
from tqdm import tqdm
import utils
import pandas as pd
import numpy as np
import librosa
import soundfile as sf
import argparse
from models import crnn
import os
import matplotlib.pyplot as plt

SAMPLE_RATE = 16000
EPS = np.spacing(1)

#新版模型将帧率改成了80
LMS_ARGS_new = {
    'n_fft': 2048,
    'n_mels': 64,
    'hop_length': int(SAMPLE_RATE * 0.0125),
    'win_length': int(SAMPLE_RATE * 0.025)
}

DEVICE = 'cpu'
if torch.cuda.is_available():
    DEVICE = 'cuda'
DEVICE = torch.device(DEVICE)


def extract_feature(wavefilepath, **kwargs):
    _, file_extension = os.path.splitext(wavefilepath)
    if file_extension == '.wav':
        wav, sr = librosa.load(wavefilepath)
    if file_extension == '.mp3':
        wav, sr = librosa.load(wavefilepath)
    elif file_extension not in ['.mp3', '.wav']:
        raise NotImplementedError('Audio extension not supported... yet ;)')
    if wav.ndim > 1:
        wav = wav.mean(-1)
    wav = librosa.resample(wav, sr, target_sr=SAMPLE_RATE)
    return np.log(
        librosa.feature.melspectrogram(wav.astype(np.float32), SAMPLE_RATE, **
                                       kwargs) + EPS).T


class OnlineLogMelDataset(torch.utils.data.Dataset):
    def __init__(self, data_list, **kwargs):
        super().__init__()
        self.dlist = data_list
        self.kwargs = kwargs

    def __getitem__(self, idx):
        return extract_feature(wavefilepath=self.dlist[idx],
                               **self.kwargs), self.dlist[idx]

    def __len__(self):
        return len(self.dlist)


MODELS = {
    'xmov': {
        'model': crnn,
        'outputdim': 2,
        'encoder': 'labelencoders/students.pth',
        'pretrained': 'xmov/hard.pt',
        'resolution': 0.0125
    },
    #支持流式的模型，单向GRU
    'xmov-s': {
        'model': crnn,
        'outputdim': 2,
        'encoder': 'labelencoders/students.pth',
        'pretrained': 'xmov/stream-hard.pt',
        'resolution': 0.0125,
        'gru_bidirection': False
    },
}

class CRNN_VAD(torch.nn.Module):
    def __init__(self, model_path, labelencoder_path, output_dim=2, gru_bidirection=True, device='cuda'):
        super().__init__()
        self.model = crnn(pretrained_from=model_path, outputdim=output_dim, gru_bidirection=gru_bidirection).to(device)
        self.resolution = 0.0125   #帧率为80
        self.labelencoder = torch.load(labelencoder_path)
        self.speech_label_idx = np.where('Speech' == self.labelencoder.classes_)[0].squeeze()
        self.sample_rate = 16000
        self.mel_nfft = 2048
        self.mel_win_length = 400  #对应16khz音频
        self.mel_hop_length = 200  #对应16khz音频，帧率为80
        self.mel_bin = 64
        self.EPS = np.spacing(1)
        self.device = device

    def forward(self, wave):
        self.model.eval()
        logmel_feat = np.log(librosa.feature.melspectrogram(wave.astype(np.float32),
                                    self.sample_rate, n_fft=self.mel_nfft, win_length=self.mel_win_length,
                                    hop_length=self.mel_hop_length, n_mels=self.mel_bin) + self.EPS).T
        with torch.no_grad():
            feature = torch.as_tensor(logmel_feat).to(self.device)
            feature = feature.unsqueeze(0)  #add one dimension to match forward
            _, vad_post = self.model(feature)
        return vad_post[0].cpu().numpy()    #single-batch

    def post_process(self, vad_post, threshold):
        if len(threshold) == 1:
            postprocessing_method = utils.binarize  # 单阈值为(0.5,)，不进行任何平滑
        else:
            postprocessing_method = utils.double_threshold  # 双阈值默认为(0.5, 0.2)进行平滑

        #speech_soft_pred = vad_post[..., self.speech_label_idx]  #选出语音帧的后验概率
        thresholded_prediction = postprocessing_method(vad_post, *threshold)
        # 将连续相同标签进行合并，得到时间戳，三个元素分别为：语音/非语音，起始时间，结束时间
        time_stamp = utils.decode_with_timestamps(self.labelencoder, thresholded_prediction)
        speech_time_stamp = [[stamp[1], stamp[2]] for stamp in time_stamp if(stamp[0]=='Speech')]
        return np.array(speech_time_stamp)*self.resolution*self.sample_rate  #将帧号转换为采样点数

    def cut_wav_in_batch(self, wave, time_stamp):
        def modify_time_stamp(time_stamp):
            max_sample_count = 0
            valid_time_stamp = []
            for stamp in time_stamp:
                new_stamp = [int(min(t, wave.shape[-1])) for t in stamp]
                max_sample_count = max(max_sample_count, new_stamp[1] - new_stamp[0])
                valid_time_stamp.append(new_stamp)
            return valid_time_stamp, max_sample_count

        time_stamp, max_sample_count = modify_time_stamp(time_stamp)
        seg_count = len(time_stamp)
        audio_seg_list = np.zeros((seg_count, max_sample_count), dtype=float)
        for i, sample_stamp in enumerate(time_stamp):
            true_length = int(sample_stamp[1]) - int(sample_stamp[0])
            audio_seg_list[i, :true_length] = wave[int(sample_stamp[0]): int(sample_stamp[1])]
        return np.array(audio_seg_list)


def plot_wav_and_vad(audio, ref, hyp, save_name):
    '''
    plot wave figure and VAD reference and hypothesis
    :param audio:
    :param ref:
    :param hyp:
    :return:
    '''
    plt.figure()
    time1 = np.arange(0, len(hyp))
    time = np.arange(0, len(audio)) * (len(hyp) / len(audio))  #将时间轴映射到标签的数量
    plt.plot(time, audio)
    if not ref is None:
        assert(len(hyp)==len(ref))
        plt.plot(time1, ref)
    plt.plot(time1, hyp)
    plt.savefig(save_name)
    plt.close()

def main():
    parser = argparse.ArgumentParser()
    group = parser.add_mutually_exclusive_group(required=True)
    group.add_argument(
        '-w',
        '--wav',
        help=
        'A single wave/mp3/flac or any other compatible audio file with soundfile.read'
    )
    group.add_argument(
        '-l',
        '--wavlist',
        help=
        'A list of wave or any other compatible audio files. E.g., output of find . -type f -name *.wav > wavlist.txt'
    )

    parser.add_argument('-model', choices=list(MODELS.keys()), default='xmov-s')
    parser.add_argument(
        '--pretrained_dir',
        default='pretrained_models',
        help=
        'Path to downloaded pretrained models directory, (default %(default)s)'
    )
    parser.add_argument('-o',
                        '--output_path',
                        default=None,
                        help='Output folder to save predictions if necessary')

    parser.add_argument('-soft',
                        default=False,
                        action='store_true',
                        help='Outputs soft probabilities.')
    parser.add_argument('-hard',
                        default=False,
                        action='store_true',
                        help='Outputs hard labels as zero-one array.')
    parser.add_argument('-th',
                        '--threshold',
                        default=(0.5, 0.2),
                        type=float,
                        nargs="+")
    args = parser.parse_args()
    pretrained_dir = Path(args.pretrained_dir)
    if not (pretrained_dir.exists() and pretrained_dir.is_dir()):
        logger.error(f"""Pretrained directory {args.pretrained_dir} not found.
Please download the pretrained models from and try again or set --pretrained_dir to your directory."""
                     )
        return
    logger.info("Passed args")
    for k, v in vars(args).items():
        logger.info(f"{k} : {str(v):<10}")
    if args.wavlist:
        wavlist = pd.read_csv(args.wavlist,
                              usecols=[0],
                              header=None,
                              names=['filename'])
        wavlist = wavlist['filename'].values.tolist()
    elif args.wav:
        wavlist = [args.wav]

    dset = OnlineLogMelDataset(wavlist, **LMS_ARGS_new)
    dloader = torch.utils.data.DataLoader(dset,
                                          batch_size=1,
                                          num_workers=3,
                                          shuffle=False)

    model_kwargs_pack = MODELS[args.model]
    model_resolution = model_kwargs_pack['resolution']
    # Load model from relative path
    model = model_kwargs_pack['model'](
        outputdim=model_kwargs_pack['outputdim'],
        pretrained_from=pretrained_dir /
        model_kwargs_pack['pretrained'],
        gru_bidirection=model_kwargs_pack.get('gru_bidirection', True)
    ).to(DEVICE).eval()
    encoder = torch.load(pretrained_dir / model_kwargs_pack['encoder'])
    logger.trace(model)

    output_dfs = []   #保存切分结果，格式为： 起始时间， 结束时间， 音频路径
    output_wav_seg_in_batch = []  #对应输入wavlist,每条音频输出切分后的音频
    output_time_stamps = []       #对应输入wavlist,每条音频输出的时间戳
    frame_outputs = {}
    frame_outputs_soft = []
    frame_outputs_hard = []
    threshold = tuple(args.threshold)

    speech_label_idx = np.where('Speech' == encoder.classes_)[0].squeeze()
    # Using only binary thresholding without filter
    if len(threshold) == 1:
        postprocessing_method = utils.binarize  #单阈值为(0.5,)，不进行任何平滑
    else:
        postprocessing_method = utils.double_threshold   #双阈值默认为(0.5, 0.2)进行平滑

    with torch.no_grad(), tqdm(total=len(dloader), leave=False, unit='clip') as pbar:
        for i, (feature,filename) in enumerate(dloader):
            feature = torch.as_tensor(feature).to(DEVICE)
            prediction_tag, prediction_time = model(feature)
            prediction_tag = prediction_tag.to('cpu')
            prediction_time = prediction_time.to('cpu')

            if prediction_time is not None:  # Some models do not predict timestamps
                cur_filename = filename[0]  #Remove batchsize
                thresholded_prediction = postprocessing_method(prediction_time, *threshold)
                speech_soft_pred = prediction_time[..., speech_label_idx]
                if args.soft:
                    speech_soft_pred = prediction_time[..., speech_label_idx].numpy()
                    frame_outputs[cur_filename] = speech_soft_pred[0]  # 1 batch
                    frame_outputs_soft.append(speech_soft_pred[0])
                if args.hard:
                    speech_hard_pred = thresholded_prediction[...,speech_label_idx]
                    frame_outputs[cur_filename] = speech_hard_pred[0]  # 1 batch
                    frame_outputs_hard.append(speech_hard_pred[0])

                #将连续相同标签进行合并，得到时间戳，三个元素分别为：语音/非语音，起始时间，结束时间
                labelled_predictions = utils.decode_with_timestamps(encoder, thresholded_prediction)
                pred_label_df = pd.DataFrame(labelled_predictions[0],columns=['event_label', 'onset', 'offset'])
                if args.output_path:
                    args.output_path = Path(args.output_path)
                    args.output_path.mkdir(parents=True, exist_ok=True)
                if not pred_label_df.empty:
                    pred_label_df['filename'] = cur_filename
                    pred_label_df['onset'] *= model_resolution
                    pred_label_df['offset'] *= model_resolution
                    pbar.set_postfix(labels=','.join(
                        np.unique(pred_label_df['event_label'].values)))
                    pbar.update()
                    output_dfs.append(pred_label_df)


                    wav_name = cur_filename.split('/')[-1].replace(".wav", "")
                    audio = librosa.load(cur_filename, sr=SAMPLE_RATE)[0]
                    pred_speech_df = pred_label_df[pred_label_df['event_label'] == 'Speech']
                    # 将语音时间戳（单位：s）保存下来
                    output_time_stamps.append(np.array(pred_speech_df[['onset', 'offset']]).tolist())
                    pred_speech_df['onset'] *= SAMPLE_RATE
                    pred_speech_df['offset'] *= SAMPLE_RATE
                    audio_sample_stamp_list = np.array(pred_speech_df[['onset', 'offset']]).tolist()

                    max_sample_count = 0
                    seg_count = len(audio_sample_stamp_list)
                    for sample_stamp in audio_sample_stamp_list:
                        max_sample_count = max(max_sample_count, int(sample_stamp[1]-sample_stamp[0])+1)
                    audio_seg_list = np.zeros((seg_count, max_sample_count), dtype=np.float)
                    for i, sample_stamp in enumerate(audio_sample_stamp_list):
                        true_length = int(sample_stamp[1])-int(sample_stamp[0])
                        audio_seg_list[i, :true_length] = audio[int(sample_stamp[0]): int(sample_stamp[1])]
                        # 输出切分之后的wav，以便检查和分析
                        if args.output_path:
                            sf.write(os.path.join(args.output_path, '{}_{:05d}.wav'.format(wav_name, i)),
                                 audio[int(sample_stamp[0]): int(sample_stamp[1])], samplerate=SAMPLE_RATE)
                    output_wav_seg_in_batch.append(audio_seg_list)

                    # 画出模型输出后验的结果和语音波形
                    if args.output_path:
                        hypothesis = speech_soft_pred[0]
                        plot_wav_and_vad(audio, None, hypothesis, f"{args.output_path}/{wav_name}.png")

    full_prediction_df = pd.concat(output_dfs)
    prediction_df = full_prediction_df[full_prediction_df['event_label'] =='Speech']
    if args.output_path:
        args.output_path = Path(args.output_path)
        args.output_path.mkdir(parents=True, exist_ok=True)
        prediction_df.to_csv(args.output_path / 'speech_predictions.tsv', sep='\t', index=False)
        full_prediction_df.to_csv(args.output_path / 'all_predictions.tsv', sep='\t', index=False)

        if args.soft or args.hard:
            prefix = 'soft' if args.soft else 'hard'
            with open(args.output_path / f'{prefix}_predictions.txt', 'w') as wp:
                np.set_printoptions(suppress=True, precision=2, linewidth=np.inf)
                for fname, output in frame_outputs.items():
                    print(f"{fname} {output}", file=wp)
        logger.info(f"Putting results also to dir {args.output_path}")
    if args.soft or args.hard:
        np.set_printoptions(suppress=True, precision=2, linewidth=np.inf)
        for fname, output in frame_outputs.items():
            print(f"{fname} {output}")
    else:
        print(prediction_df.to_markdown(showindex=False))

    return output_wav_seg_in_batch, output_time_stamps

if __name__ == "__main__":
    output_wav_seg_in_batch, output_time_stamps = main()
    pprint.pprint(output_time_stamps)


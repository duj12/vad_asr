import torch
from loguru import logger
from pathlib import Path
from tqdm import tqdm
import utils
import numpy as np
import librosa
import soundfile as sf
from models import crnn
import os
import matplotlib.pyplot as plt

SAMPLE_RATE = 16000
DEVICE = 'cpu'
if torch.cuda.is_available():
    DEVICE = 'cuda'
DEVICE = torch.device(DEVICE)

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
        return np.array(speech_time_stamp)*self.resolution*self.sample_rate, thresholded_prediction   #将帧号转换为采样点数

    def cut_wav_in_batch(self, wave, time_stamp):
        max_sample_count = 0
        seg_count = len(time_stamp)
        for sample_stamp in time_stamp:
            max_sample_count = max(max_sample_count, int(sample_stamp[1]) - int(sample_stamp[0]) )
        audio_seg_list = np.zeros((seg_count, max_sample_count), dtype=float)
        for i, sample_stamp in enumerate(time_stamp):
            true_length = int(sample_stamp[1]) - int(sample_stamp[0])
            audio_seg_list[i, :true_length] = wave[int(sample_stamp[0]): int(sample_stamp[1])]
        return np.array(audio_seg_list)

class CRNN_VAD_STREAM(torch.nn.Module):
    def __init__(self, model_path, labelencoder_path, output_dim=2, gru_bidirection=False, device='cuda'):
        super().__init__()
        self.model = crnn(pretrained_from=model_path,
                          outputdim=output_dim,
                          gru_bidirection=gru_bidirection).to(device)
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

        self.model.eval()
        #because of the exist of 4-time downsample in CNN, the frame context is best to be 4*N
        #CNN Reception Field=11, left and right context best to be larger than 11
        self.left_context = 16  #5-Conv2d layers, each layer with 1-frame padding,
        self.central_context = 48
        self.right_context = 16   #5-Conv2d layers, each layer with 1-frame padding,

        self.chunk_length = 4096  #~250ms per chunk, must bigger than self.mel_nfft, best to be N*self.mel_nfft

    def extract_mel_online(self, wave_chunk, wave_remained):
        #首先根据前一个chunk剩下的采样点数，计算这个chunk的实际帧数以及剩余的采样点数
        if not wave_remained is None:
            wave_chunk = np.append(wave_remained, wave_chunk)
        chunk_len = wave_chunk.shape[-1]
        n_frames = 1+(chunk_len-self.mel_nfft)//self.mel_hop_length
        #剩下的采样点为当前chunk最后一帧起始位置之后hop_len起的采样点
        remained_pos = n_frames * self.mel_hop_length
        wave_remained = wave_chunk[remained_pos:]
        log_mel = np.log(
            librosa.feature.melspectrogram(wave_chunk.astype(np.float32), SAMPLE_RATE,
                                           n_fft=self.mel_nfft, win_length=self.mel_win_length,
                                           hop_length=self.mel_hop_length, n_mels=self.mel_bin,
                                           center=False) + self.EPS).T
        return log_mel, wave_remained


    def forward(self, wave):
        '''
        :param wave: In this demo we load all samples, in this method we split into chunks to simulate streaming.
        :return: vad probability
        '''
        wav_len = wave.shape[-1]
        vad_post = np.empty((0, 2), dtype=wave.dtype)
        frames_queue = np.empty((0, self.mel_bin), dtype=wave.dtype)
        wave_remained = None
        hidden_vector = None   # GRU hidden buffer
        first_chunk, last_chunk = False, False
        for chunk_id, start in enumerate(range(0, wav_len, self.chunk_length)):
            end = min(start+self.chunk_length, wav_len)
            wave_chunk = wave[start:end]
            if chunk_id == 0:
                first_chunk = True
                #padding in the begining, must be exactly the same as offline fashion
                wave_chunk = np.pad(wave_chunk, (self.mel_nfft//2, 0), mode='reflect')
            if start+self.chunk_length >= wav_len:
                last_chunk = True
                #padding in the end
                wave_chunk = np.pad(wave_chunk, (0, self.mel_nfft//2), mode='reflect')
            logmel_feat, wave_remained = self.extract_mel_online(wave_chunk, wave_remained)
            #特征入队
            frames_queue = np.append(frames_queue, logmel_feat, axis=0)

            #为了可读性，这里将last_chunk, first_chunk, middle_chunk分开处理
            #last chunk, input l+c+r, output c+r
            if last_chunk:
                cur_frame_chunk = frames_queue
                with torch.no_grad():
                    feature = torch.as_tensor(cur_frame_chunk).to(self.device)
                    feature = feature.unsqueeze(0)  # add one dimension to match forward
                    _, time, _ = feature.shape
                    cnn_feature = self.model.forward_stream_vad_cnn(feature)
                    _, down_time, _ = cnn_feature.shape
                    down_sample_factor = time // down_time  # 4-time down_sampling
                    # since we only output central+right, we should keep the context vector at position of central+right
                    cnn_feature_to_gru = cnn_feature[:, self.left_context//down_sample_factor:, :]
                    gru_output, hidden_vector = self.model.forward_stream_vad_gru(cnn_feature_to_gru, hidden_vector)
                    vad_chunk = self.model.forward_stream_vad_upsample(gru_output, time-self.left_context)
                    vad_chunk = vad_chunk[0].cpu().numpy()  # single-batch
                    vad_post = np.append(vad_post, vad_chunk, axis=0)
                break
            # first_chunk is processed, input l+c+r, output l+c
            if first_chunk and (frames_queue.shape[0] > self.left_context+self.central_context+self.right_context):
                cur_frame_chunk = frames_queue[:self.left_context + self.central_context + self.right_context]
                with torch.no_grad():
                    feature = torch.as_tensor(cur_frame_chunk).to(self.device)
                    feature = feature.unsqueeze(0)  # add one dimension to match forward
                    _, time, _ = feature.shape
                    cnn_feature = self.model.forward_stream_vad_cnn(feature)
                    _, down_time, _ = cnn_feature.shape
                    down_sample_factor = time // down_time  #4-time down_sampling
                    # since we only output left+central, we should keep the context vector at position of left+central
                    cnn_feature_to_gru = cnn_feature[:, :(self.left_context + self.central_context)//down_sample_factor, :]
                    gru_output, hidden_vector = self.model.forward_stream_vad_gru(cnn_feature_to_gru, hidden_vector)
                    vad_chunk = self.model.forward_stream_vad_upsample(gru_output, self.left_context+self.central_context)
                    vad_chunk = vad_chunk[0].cpu().numpy()  # single-batch
                    vad_post = np.append(vad_post, vad_chunk, axis=0)
                frames_queue = frames_queue[self.central_context:]  # 队首central_context之前帧出队
                first_chunk = False
            #middle chunk. input l+c+r, output c
            while frames_queue.shape[0] > self.left_context+self.central_context+self.right_context:
                cur_frame_chunk = frames_queue[:self.left_context+self.central_context+self.right_context]
                with torch.no_grad():
                    feature = torch.as_tensor(cur_frame_chunk).to(self.device)
                    feature = feature.unsqueeze(0)  #add one dimension to match forward
                    _, time, _ = feature.shape
                    cnn_feature = self.model.forward_stream_vad_cnn(feature)
                    _, down_time, _ = cnn_feature.shape
                    down_sample_factor = time//down_time
                    #since we only output central, we should keep the context vector at position of central
                    cnn_feature_to_gru = cnn_feature[:, self.left_context//down_sample_factor:
                                            (self.left_context + self.central_context) //down_sample_factor, :]
                    gru_output, hidden_vector = self.model.forward_stream_vad_gru(cnn_feature_to_gru, hidden_vector)
                    vad_chunk = self.model.forward_stream_vad_upsample(gru_output, self.central_context)
                    vad_chunk = vad_chunk[0].cpu().numpy()  #single-batch
                    vad_post = np.append(vad_post, vad_chunk, axis=0)
                frames_queue = frames_queue[self.central_context:]  #队首central_context之前帧出队
        return vad_post


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
        return np.array(speech_time_stamp)*self.resolution*self.sample_rate, thresholded_prediction  #将帧号转换为采样点数

    def cut_wav_in_batch(self, wave, time_stamp):
        def modify_time_stamp(time_stamp):
            max_sample_count = 0
            valid_time_stamp = []
            for stamp in time_stamp:
                new_stamp = [int(min(t, wave.shape[-1])) for t in stamp]
                max_sample_count = max(max_sample_count, new_stamp[1]-new_stamp[0])
                valid_time_stamp.append(new_stamp)
            return valid_time_stamp, max_sample_count
        time_stamp, max_sample_count= modify_time_stamp(time_stamp)
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

def measure_vad_performance(reference,prediction, hypothesis):
    '''

    :param reference: vad reference, 0-sil, 1-speech
    :param prediction: thresholded output, 0-sil, 1-speech
    :param hypothesis: probability of speech
    :return:
    '''
    assert (len(reference) == len(hypothesis))
    logger.info("Calculating VAD measures ... ")
    import metrics
    speech_frame_ground_truth = np.concatenate(reference, axis=0)
    speech_frame_predictions = np.concatenate(prediction, axis=0)
    speech_frame_prob_predictions = np.concatenate(hypothesis, axis=0)
    vad_results = []
    tn, fp, fn, tp = metrics.confusion_matrix(
        speech_frame_ground_truth, speech_frame_predictions).ravel()
    fer = 100 * ((fp + fn) / len(speech_frame_ground_truth))
    acc = 100 * ((tp + tn) / (len(speech_frame_ground_truth)))
    p_miss = 100 * (fn / (fn + tp))
    p_fa = 100 * (fp / (fp + tn))
    for i in [0.1, 0.3, 0.5, 0.7, 0.9]:
        mp_fa, mp_miss = metrics.obtain_error_rates(
            speech_frame_ground_truth, speech_frame_prob_predictions, i)
        tn, fp, fn, tp = metrics.confusion_matrix(
            speech_frame_ground_truth,
            speech_frame_prob_predictions > i).ravel()
        sub_fer = 100 * ((fp + fn) / len(speech_frame_ground_truth))
        logger.info(
            f"PFa {100 * mp_fa:.2f} Pmiss {100 * mp_miss:.2f} FER {sub_fer:.2f} t: {i:.2f}"
        )

    auc = metrics.roc(speech_frame_ground_truth, speech_frame_prob_predictions) * 100
    for avgtype in ('micro', 'macro', 'binary'):
        precision, recall, f1, _ = metrics.precision_recall_fscore_support(
            speech_frame_ground_truth,
            speech_frame_predictions,
            average=avgtype)
        vad_results.append((avgtype, 100 * precision, 100 * recall, 100 * f1))
    for avgtype, precision, recall, f1 in vad_results:
        print(
            f"VAD {avgtype:<10} F1: {f1:<10.3f} Pre: {precision:<10.3f} Recall: {recall:<10.3f}"
        )
    print(f"FER: {fer:.2f}")
    print(f"AUC: {auc:.2f}")
    print(f"Pfa: {p_fa:.2f}")
    print(f"Pmiss: {p_miss:.2f}")
    print(f"ACC: {acc:.2f}")


if __name__ == "__main__":
    #vad_ckpt_path =  'pretrained_models/xmov/hard.pt'  #离线模型
    vad_ckpt_path = 'pretrained_models/xmov/stream-hard.pt'  #流式模型
    labelencoder_path = 'pretrained_models/labelencoders/students.pth'
    labelencoder = torch.load(labelencoder_path)
    vad_model = CRNN_VAD_STREAM(vad_ckpt_path, labelencoder_path)

    #测试vad性能
    threshold = (0.5,)   #(0.5,), (0.5, 0.2)
    save_cutted_wav = False
    wav_list = '/data/megastore/Projects/DuJing/code/Datadriven-VAD/data/test/test.list'
    ref_list = '/data/megastore/Projects/DuJing/code/Datadriven-VAD/data/test/test.vad'
    output_path = '/data/megastore/Projects/DuJing/code/Datadriven-VAD/data/test/XmovVAD_stream0/test'

    # #实际切分音频
    # threshold = (0.5, 0.2)
    # save_cutted_wav = True
    # wav_list = 'wav.list'
    # ref_list = None
    # output_path = 'output_stream'

    f_wav = open(wav_list, 'r')
    if ref_list:
        f_ref = open(ref_list, 'r')
    hypothesis = []
    reference = []
    prediction = []
    for line in f_wav.readlines():
        wav_path = line.strip()
        logger.info(f'processing {wav_path}')

        _, file_extension = os.path.splitext(wav_path)
        if file_extension == '.wav':
            wave, sr = librosa.load(wav_path)
        if file_extension == '.mp3':
            wave, sr = librosa.load(wav_path)
        elif file_extension not in ['.mp3', '.wav']:
            raise NotImplementedError('Audio extension not supported... yet ;)')
        if wave.ndim > 1:
            wave = wave.mean(-1)
        if sr != 16000:
            wave = librosa.resample(wave, sr, target_sr=16000)

        vad_post = vad_model.forward(wave)
        time_stamp, hard_pred = vad_model.post_process(vad_post, threshold=threshold)
        #cutted_wave = vad_model.cut_wav_in_batch(wave, time_stamp)  #in some case we need cutted wav in batch

        speech_prob = vad_post[:, 1]
        pred_label = hard_pred[:, 1]
        hypothesis.append(speech_prob)
        prediction.append(pred_label)
        wav_name = wav_path.split('/')[-1].replace(".wav", "")
        if output_path:
            output_path = Path(output_path)
            output_path.mkdir(parents=True, exist_ok=True)
        if save_cutted_wav:
            for i, sample_stamp in enumerate(time_stamp):
                true_length = int(sample_stamp[1]) - int(sample_stamp[0])
                # 输出切分之后的wav，以便检查和分析
                if output_path:
                    sf.write(os.path.join(output_path, '{}_{:05d}.wav'.format(wav_name, i)),
                             wave[int(sample_stamp[0]): int(sample_stamp[1])], samplerate=SAMPLE_RATE)

        if ref_list:
            # 提供了参考标注，以及预测hard标签
            line = f_ref.readline()
            hyp = speech_prob
            ref = np.array([int(x) for x in line.strip().split(' ')])
            hyp_sz = (hyp.shape)[0]
            ref_sz = (ref.shape)[0]
            ref2hyp_ratio = float(ref_sz / hyp_sz)
            if (hyp_sz != ref_sz):  # 对标签进行采样
                target_inds = (np.arange(hyp_sz) * ref2hyp_ratio).astype(int)
                # 这里为了避免对齐之后的标签越界，将超过标签长度的idx替换成最后一个标签序号
                target_inds[target_inds >= ref_sz] = ref_sz - 1
                ref = ref[target_inds]
            assert (hyp.size == ref.size)
            reference.append(ref)
            plot_wav_and_vad(wave, ref, hyp, f"{output_path}/{wav_name}.png")
        # 画出模型输出后验的结果和语音波形
        else:
            plot_wav_and_vad(wave, None, speech_prob, f"{output_path}/{wav_name}.png")
    if ref_list:
        measure_vad_performance(reference, prediction, hypothesis)
    f_wav.close()
    if ref_list:
        f_ref.close()
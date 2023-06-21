import os
import sys
from time import time

import librosa
import numpy as np
import soundfile as sf
import torch
from loguru import logger
from vad import CRNN_VAD_STREAM


def vad_process(wav_list, output_path):
    sample_rate = 16000
    device = 'cpu'  # cpu is fast enough.
    if torch.cuda.is_available():
        device = 'cuda'

    vad_model = CRNN_VAD_STREAM(left_frames=12,
                                right_frames=12,
                                device=device)

    save_cutted_wav = True
    if not os.path.exists(output_path):
        os.mkdir(output_path)

    total_time = 0

    for f_name in wav_list:
        f_name=f_name.strip()
        logger.info(f'processing {f_name}')
        wav_name = f_name.strip().split('/')[-1].split('.')[0]
        wave, sr = librosa.load(f_name)
        if wave.ndim > 1:
            wave = wave.mean(-1)
        if sr != 16000:
            wave = librosa.resample(wave, orig_sr=sr, target_sr=16000)

        # clear GRU buffer at the begining of each wave processing
        vad_model.clear_gru_buffer()
        speech_segment = []    # 保存过滤掉噪音和静音的音频
        speech_segment_num = 0
        asr_segment = []       # 保存用于ASR的断句之后的音频
        asr_segment_num = 0

        wave_padded = wave
        # # 下面两个pad在实际流式场景中不会去做，这里是为了验证前向计算的结果，按照和训练时特征提取保持一样的操作。
        # # 这里多pad了mel_nfft + right_samples 个采样点，实际上就是模型的时延
        # wave_padded = np.pad(wave, (vad_model.mel_nfft // 2), mode='reflect')
        # # 提前给音频 pad 上后文，实际上不会输出这部分，只会输出中间部分
        # wave_padded = np.pad(wave_padded, (0, vad_model.right_samples))

        wav_padded_len = wave_padded.shape[-1]

        start_at = time()

        # 每次传入的采样点数需为200的整数倍，这里假定每次传入0.3s采样率为16k的音频，即4800个采样点
        central_samples = 4800
        for start in range(0, wav_padded_len, central_samples):
            end = min(start + central_samples, wav_padded_len)

            wave_chunk = wave_padded[start:end]
            # wave left is less than buffer length
            if wav_padded_len - start < len(vad_model.buf):
                wave_chunk = wave_padded[start:]
                start = wav_padded_len  # to break the loop

            # 计算语音概率，当前chunk状态，语音标签，ASR断句类型
            vad_post_chunk, chunk_state, vad_pred_chunk, asr_endpoint = \
                vad_model.stream_asr_endpoint(wave_chunk,
                                              max_speech_len=15,
                                              max_trailing_silence=500,)

            # # 下面保存的音频是去除了静音/噪音的音频，只剩下应该被激活的音频
            # if chunk_state != 0:  # 只要不是整段静音就拼起来
            #     speech_segment += wave_chunk.tolist()
            # if chunk_state == 3:  # SPEECH_END
            #     # 只要一段音频结束，就输出切分之后的 wav，以便检查和分析
            #     if output_path and save_cutted_wav:
            #         sf.write(os.path.join(
            #             output_path,
            #             f"{wav_name}_active_{speech_segment_num:05d}.wav"),
            #                  np.array(speech_segment),
            #                  samplerate=sample_rate)
            #     speech_segment_num += 1
            #     speech_segment = []  # clear the segment

            # 下面演示用于ASR分句的VAD效果，保存的音频只是在流式输入的基础上根据需要断开的位置进行了断句
            asr_segment += wave_chunk.tolist()
            if asr_endpoint > 0  and len(asr_segment) >= 5*16000: # vad检测到endpoint
                vad_model.reset_state()  # 需要重置，以便进行下一次的检测
                # 只要一段音频结束，就输出切分之后的 wav，以便检查和分析
                if output_path and save_cutted_wav:
                    sf.write(os.path.join(
                        output_path,
                        f"{wav_name}_{asr_segment_num:05d}.wav"),
                        np.array(asr_segment),
                        samplerate=sample_rate)
                asr_segment_num += 1
                asr_segment = []  # clear the segment

            if start == wav_padded_len:
                break

        end_at = time()
        logger.info(f"{f_name} {end_at - start_at}s")
        total_time += (end_at - start_at)

    logger.info(f"device: {device}, total_time: {total_time}s")

if __name__ == "__main__":
    wav_list = sys.argv[1]
    output_path = sys.argv[2]
    MAX_THREAD=32

    f_wav = open(wav_list, 'r')
    file_list = []
    for i, line in enumerate(f_wav.readlines()):
        file_list.append(line)

    th_cnt = min(MAX_THREAD, len(file_list))

    if th_cnt==1:
        vad_process(file_list, output_path)
    else:
        import threading
        total_cnt = len(file_list)
        job_per_th = total_cnt // th_cnt if(total_cnt%th_cnt==0) else total_cnt // th_cnt + 1
        threads = []
        for th in range(th_cnt):
            start_idx = th * job_per_th
            end_idx = min((th+1) * job_per_th, total_cnt)
            threads.append(threading.Thread(target=vad_process, args=(file_list[start_idx:end_idx], output_path)))
        for th in range(th_cnt):
            threads[th].start()
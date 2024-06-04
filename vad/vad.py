from typing import Tuple

import librosa
import numpy as np
import torch

from .crnn import crnn
from .openvino_feathures import Features


def binarize(pred, threshold=0.5):

    def thresholded(soft_prob, threshold=0.5):
        high_idx = soft_prob >= threshold
        low_idx = soft_prob < threshold
        soft_prob[high_idx] = 1.0
        soft_prob[low_idx] = 0.0
        return soft_prob

    # Batch_wise
    if pred.ndim == 3:
        return np.array(
            [thresholded(sub, threshold=threshold) for sub in pred])
    else:
        return thresholded(pred, threshold=threshold)


def double_threshold(x, high_thres, low_thres, n_connect=1):
    """double_threshold
    Helper function to calculate double threshold for n-dim arrays

    :param x: input array
    :param high_thres: high threshold value
    :param low_thres: Low threshold value
    :param n_connect: Distance of <= n clusters will be merged
    """
    assert x.ndim <= 3, "Whoops something went wrong with the input ({}), check if its <= 3 dims".format(
        x.shape)
    if x.ndim == 3:
        apply_dim = 1
    elif x.ndim < 3:
        apply_dim = 0
    # x is assumed to be 3d: (batch, time, dim)
    # Assumed to be 2d : (time, dim)
    # Assumed to be 1d : (time)
    # time axis is therefore at 1 for 3d and 0 for 2d (
    return np.apply_along_axis(lambda x: _double_threshold(
        x, high_thres, low_thres, n_connect=n_connect),
                               axis=apply_dim,
                               arr=x)


def find_contiguous_regions(activity_array):
    """Find contiguous regions from bool valued numpy.array.
    Copy of https://dcase-repo.github.io/dcase_util/_modules/dcase_util/data/decisions.html#DecisionEncoder

    Reason is:
    1. This does not belong to a class necessarily
    2. Import DecisionEncoder requires sndfile over some other imports..which causes some problems on clusters

    """

    # Find the changes in the activity_array
    change_indices = np.logical_xor(activity_array[1:],
                                    activity_array[:-1]).nonzero()[0]

    # Shift change_index with one, focus on frame after the change.
    change_indices += 1

    if activity_array[0]:
        # If the first element of activity_array is True add 0 at the beginning
        change_indices = np.r_[0, change_indices]

    if activity_array[-1]:
        # If the last element of activity_array is True, add the length of the array
        change_indices = np.r_[change_indices, activity_array.size]

    # Reshape the result into two columns
    return change_indices.reshape((-1, 2))


def connect_(pairs, n=1):
    """connect_
    Connects two adjacent clusters if their distance is <= n

    :param pairs: Clusters of iterateables e.g., [(1,5),(7,10)]
    :param n: distance between two clusters
    """
    if len(pairs) == 0:
        return []
    start_, end_ = pairs[0]
    new_pairs = []
    for i, (next_item, cur_item) in enumerate(zip(pairs[1:], pairs[0:])):
        end_ = next_item[1]
        if next_item[0] - cur_item[1] <= n:
            pass
        else:
            new_pairs.append((start_, cur_item[1]))
            start_ = next_item[0]
    new_pairs.append((start_, end_))
    return new_pairs


def connect_clusters_(x, n=1):
    """connect_clusters_
    Connects clustered predictions (0,1) in x with range n

    :param x: Input array. zero-one format
    :param n: Number of frames to skip until connection can be made
    """
    assert x.ndim == 1, "input needs to be 1d"
    reg = find_contiguous_regions(x)
    start_end = connect_(reg, n=n)
    zero_one_arr = np.zeros_like(x, dtype=int)
    for sl in start_end:
        zero_one_arr[sl[0]:sl[1]] = 1
    return zero_one_arr


def _double_threshold(x, high_thres, low_thres, n_connect=1, return_arr=True):
    """_double_threshold
    Computes a double threshold over the input array

    :param x: input array, needs to be 1d
    :param high_thres: High threshold over the array
    :param low_thres: Low threshold over the array
    :param n_connect: Postprocessing, maximal distance between clusters to connect
    :param return_arr: By default this function returns the filtered indiced, but if return_arr = True it returns an array of tsame size as x filled with ones and zeros.
    """
    assert x.ndim == 1, "Input needs to be 1d"
    high_locations = np.where(x > high_thres)[0]
    locations = x > low_thres
    encoded_pairs = find_contiguous_regions(locations)

    filtered_list = list(
        filter(
            lambda pair:
            ((pair[0] <= high_locations) & (high_locations <= pair[1])).any(),
            encoded_pairs))

    filtered_list = connect_(filtered_list, n_connect)
    if return_arr:
        zero_one_arr = np.zeros_like(x, dtype=int)
        for sl in filtered_list:
            zero_one_arr[sl[0]:sl[1]] = 1
        return zero_one_arr
    return filtered_list


def connect_clusters(x, n=1):
    if x.ndim == 1:
        return connect_clusters_(x, n)
    if x.ndim >= 2:
        return np.apply_along_axis(lambda a: connect_clusters_(a, n=n), -2, x)


def decode_with_timestamps(labels):
    """decode_with_timestamps
    Decodes the predicted label array (2d) into a list of
    [(Labelname, onset, offset), ...]

    :param encoder: Encoder during training
    :type encoder: pre.MultiLabelBinarizer
    :param labels: n-dim array
    :type labels: np.array
    """
    if labels.ndim == 2:
        return [_decode_with_timestamps(lab) for lab in labels]
    else:
        return _decode_with_timestamps(labels)


def _decode_with_timestamps(labels):
    change_indices = find_contiguous_regions(labels)

    return change_indices


class CRNN_VAD_STREAM(torch.nn.Module):
    '''
    Online VAD module, with 5-CNN + GRU.
    Because of the existing of 4-time down-sampling in CNN,
    the frame context is best to be 4*N.
    The total reception field=11, left and right frames should be larger than 11
    chunk_length(sample counts) = (total_frame + 10) * 200
    '''

    def __init__(self,
                 left_frames=12,
                 right_frames=12,
                 device='cpu'):
        """
        :param left_frames: 目标推理帧左侧参考帧数，要求 >= 12，最好为 16。对一段音频进
            行推理时，需要参考这段音频前后的音频信息。对于 16khz 的音频，本模型将一秒的音频
            划分为 80 帧，一帧对应 200 个采样点。
        :param right_frames: 目标推理帧右侧的参考帧数，要求 >= 12，最好为 16。含义和
            left_frames 类似。
        :param device: 该模型运行的设备。合法值包括 "cpu"、"cuda"。
        """
        super().__init__()
        self.resolution = 0.0125  # 帧率为 80，每帧对应0.0125s
        self.speech_label_idx = np.array(1)  # 0: None-Speech, 1: Speech
        self.sample_rate = 16000
        self.mel_nfft = 2048       # 实际特征提取分帧时每一帧的长度
        self.mel_win_length = 400  #
        self.mel_hop_length = 200  # 帧移，对应 16khz 音频，帧率为 80
        self.mel_bin = 64
        self.EPS = np.spacing(1)
        self.device = device
        self.cnn_subsample_factor = 4

        self.left_frames = left_frames    #推荐值12
        self.right_frames = right_frames  #推荐值12，越小时延越小，但性能会变差

        self.left_samples = self.left_frames * self.mel_hop_length
        self.right_samples = self.right_frames * self.mel_hop_length
        self.nfft2hoplen = self.mel_nfft // self.mel_hop_length  # 10，这个是固定时延，由特征提取引入
        self.latency_frames = self.right_frames + self.nfft2hoplen   # 总的时延帧数

        self.gru_vector = None   # GRU 网络累积的历史向量

        # vad_judgement related
        self.state = 0  # 0:SIL, 1:SPEECH_START, 2:SPEECH_ON, 3:SPEECH_END
        self.max_speech_length = 20  # 限制最长语音段长度，单位为s, 默认为20s。
        self.max_silence_length = 5  # 连续静音段最长长度，超过此值音频被截断。单位s，默认5s。
        self.max_trailing_silence = 1000   # 拖尾静音参数，单位为ms, 默认1000ms
        self.sil_frames = 0  # 进入 SPEECH_END或 SIL 状态之后已经累计的静音帧
        self.total_frames = 0  # 上一次切分之后，已经累积的总语音帧
        self.has_speech = False   # 当前这一段音频是否含有语音。如果不含，那么静音超过5s才断开，如果包含，那么拖尾静音超过1s就断开
        self.endpoint_state = 0

        self.model = crnn(inputdim=64,
                          left_samples=self.left_samples,
                          right_samples=self.right_samples,
                          down_sample_factor=self.mel_hop_length *
                          self.cnn_subsample_factor,
                          mel_hop_length=self.mel_hop_length,
                          gru_bidirection=False,
                          device=device)
        self.model.eval()
        if device == "cpu":
            self.model.features = Features()
        # 缓存
        self.buf = np.zeros(
            ((self.left_frames + self.right_frames + self.nfft2hoplen) *
             self.mel_hop_length, ), np.float32)

    def clear_gru_buffer(self):
        self.gru_vector = None
        self.buf = np.zeros(
            ((self.left_frames + self.right_frames + self.nfft2hoplen) *
             self.mel_hop_length, ), np.float32)

    def reset_state(self):
        self.state = 0
        self.endpoint_state = 0
        self.has_speech = False
        self.total_frames = 0
        self.sil_frames = 0

    def stream_judge(self, wave: np.ndarray) -> int:
        """流式检测音频静音情况.

        特别注意：
        模型在推理某段音频时需要参考这段音频之前的音频和之后的音频，称为前向参考和后向参考。
        前向参考的帧数和后向参考的帧数由初始化模型时的 left_frames 和 right_frames 参数
        决定。另外，模型的内部实现特性要求在后向参考帧之后还必须存在 10 帧保留数据。每帧对应
        mel_hop_length 个采样点，目前 mel_hop_length == 200。

        所以每次调用时，wave 尾部的 (right_frames+10) * mel_hop_length 个采样点并不
        是推理目标，而是用作后向参考和保留数据。而用作前向参考的 left_frames 帧数据则来自
        上次调用时缓存的数据。

        对于当前调用，wave 尾部的 right_frames+10 帧数据没有被推理，所以需要将其缓存下
        来，留到下次调用时再推理。下次推理时，还是需要 left_frames 帧数据作为前向参考，所以
        最终需要缓存的数据为 wave 尾部的 left_frames + right_frames + 10 帧，即
        wave[-(left_frames + right_frames + 10) * mel_hop_length:] 被缓存。

        假设上一次传入的参数为 wave0，本次传入的参数为 wave，依据上面的说的，本次推理的目标
        为 wave0[-(right_frames+10) * mel_hop_length:] +
        wave[:-(right_frames+10) * mel_hop_length]，推理目标的长度等于 len(wave)。

        首次调用时，缓存为空白数据，假装首次调用之前都是静音。

        由于每次调用时，wave 尾部的 right_frames+10 帧数据没有被推理，所以返回的结果存
        在 right_frames+10 帧的偏移（或者叫延时）。本模型假设音频为 16khz，每秒音频分为
        为 80 帧，即每帧对于 200 个采样点。所以偏移时长为 (right_frames+10) / 80 秒。
        如果 right_frames == 12，可算出偏移时长为 150 毫秒。

        :param wave，numpy array（维度为 1，dtype 为 float32）表示的一段音频，要求音
            频为 16khz 单声道。长度需为200的整数倍

        :return：一个 int 类型的状态值。取值含义如下。
            0：整个 wave 都是静音，或者只有中间出现过很短的语音。
            1：wave 从静音变成了语音，也就是 wave 开始是静音，结尾是语音。
            2：wave 的开始和结尾都是语音。
            3：wave 从语音变成了静音，也就是开头是语音，结尾是静音。
        """

        # 需要将上次缓存的数据和 wave 前部分拼接起来再推理
        wave = np.append(self.buf, wave)   # 避免输入wav_chunk低于buf长度
        _, state, _, _ = self.forward(wave)
        # 更新缓存
        self.buf = wave[-len(self.buf):]

        return state

    def stream_asr_endpoint(
            self,
            wave: np.ndarray,
            max_speech_len: int = 20,
            max_trailing_silence: int = 1000
    ) -> Tuple[np.ndarray, np.ndarray, int]:
        """逻辑和 stream_judge 方法一样，只是为了验证，多返回了一些数据. """
        self.max_speech_length = max_speech_len
        self.max_trailing_silence = max_trailing_silence

        wave = np.append(self.buf, wave)
        speech_prob, state, label, endpoint = self.forward(wave)
        # 更新缓存
        self.buf = wave[-len(self.buf):]

        return speech_prob, state, label, endpoint

    def forward(self, wave: np.ndarray,
                threshold = (0.5, 0.2),
                ):
        """推理 wave 音频的语音和静音情况。要求音频为 16khz、单声道.

        :param wave: chunked wave samples.表示音频采样数据的一维 numpy array, dytpe
            为 float32。特别要注意 wave 的长度，需要是200的整数倍。
        :param threshold: 用于将VAD概率转成0/1的阈值，默认为双阈值，也可以设为简单的单阈值

        :return: 元组，tuple(speech_prob, state, label, endpoint)
            speech_prob：一个numpy 数组，dytpe 为 float32。长度即为当前chunk输入语音的帧数。
                数值表示每一帧是语音的概率。
            state: wave 的状态值。各取值含义如下。
                0：整个 wave 都是静音，或者只有中间出现过很短的语音。
                1：wave 从静音变成了语音，也就是 wave 开始是静音，结尾是语音。
                2：wave 的开始和结尾都是语音。
                3：wave 从语音变成了静音，也就是开头是语音，结尾是静音。
            label: 对vad输出的后验概率进行后处理之后的0/1标签预测值。0表示非语音，1表示语音
            endpoint: 当前chunk是否为endpoint, 范围值0-3，意义如下：
                0： 不是endpoint，不需要断句。
                1： 是endpoint，由于静音长度超过阈值引起
                2： 是endpoint，由于拖尾静音超过阈值引起
                3： 是endpoint，由于总音频长度超过阈值引起

        """
        # step1. 特征提取
        logmel_feat = self.extract_mel_online(wave)

        # step2. 模型前向计算，得到软标签，即概率值, 第0列表示静音概率，第1列表示语音概率
        with torch.no_grad():
            feature = torch.as_tensor(logmel_feat).to(self.device)
            feature = feature.unsqueeze(
                0)  # add one dimension to match forward
            _, time, _ = feature.shape
            output_frame_num = time - (self.left_frames + self.right_frames)
            cnn_feature = self.model.forward_stream_vad_cnn(feature)

            # since we only output central, we should keep the context vector at position of central
            cnn_feature_to_gru = cnn_feature[:, self.left_frames // self.cnn_subsample_factor:
                                                (time-self.right_frames) // self.cnn_subsample_factor, :]
            gru_output, self.gru_vector = self.model.forward_stream_vad_gru(
                cnn_feature_to_gru, self.gru_vector)
            vad_chunk = self.model.forward_stream_vad_upsample(
                gru_output, output_frame_num)
            vad_post = vad_chunk[0].cpu().numpy()  # single-batch

        speech_prob = vad_post[:, 1]  # 取出第一列表示的是语音的概率

        # step3. 后处理得到硬标签，第0列表示静音，第1列表示语音
        hard_pred = self.post_process(vad_post, threshold=threshold,)

        # step4. 取出语音标签进行判决，得到当前wav chunk的状态, 同时会对标签进行一定的平滑
        state, label = self.vad_judgement(hard_pred[:, 1])

        # step5. 由当前状态和输入label, 以及累积的历史信息，进行断句
        endpoint = self.vad_endpoint(label)

        return speech_prob, state, label, endpoint

    def extract_mel_online(self, wave_chunk):
        """
        because mel_nfft=2048, in each chunk, when extract_mel_feat, we remain 2000 samples in each process.
        :param wave_chunk: the input wave.
        :return:
        """
        log_mel = np.log(
            librosa.feature.melspectrogram(y=wave_chunk.astype(np.float32),
                                           sr=self.sample_rate,
                                           n_fft=self.mel_nfft,
                                           win_length=self.mel_win_length,
                                           hop_length=self.mel_hop_length,
                                           n_mels=self.mel_bin,
                                           center=False) + self.EPS).T
        return log_mel

    def post_process(self,
                     vad_post,
                     threshold=(0.5, 0.2)):
        '''
        :param vad_post: vad probability
        :param threshold: a tuple, with one or two elements. Default in double threshold.
        :return:
                thresholded_prediction:经过阈值处理之后的每帧的 VAD 结果，也就是将
                vad probability 中的小数全都变成了整数 0 或者 1。
        '''
        if len(threshold) == 1:
            # 单阈值为(0.5,)，不进行任何平滑
            postprocessing_method = binarize
        else:
            # 双阈值默认为 (0.5, 0.2) 进行平滑
            postprocessing_method = double_threshold

        thresholded_prediction = postprocessing_method(vad_post, *threshold)

        return thresholded_prediction.astype(int)

    def vad_judgement(
            self,
            post_label,
            off_on_length=4,
            on_off_length=4,
            hang_before=4,
            hang_over=4) -> Tuple[np.ndarray, int]:
        """
        Given input wave_chunk, output the real speech in wave according to the VAD-label.
        And return the state of current chunk. The state could be the following 4 types:
        SIL, SPEECH_START, SPEECH_ON, SPEECH_END.
        适用于不低于20帧的chunk，chunk长度越长，效果越好

        :param post_label: the label that VAD model give, in frame.
        :param off_on_length: from sil to speech, we need how many speech frames.
        :param on_off_length: from speech to sil, we need how many sil frames.
        :param hang_before: if speech onset is detection, we add how many speech frames before the onset point.
        :param hang_over: if speech offset is detection, we add how many speech frames after the offset point.
        :return:
            output_speech: wave_chunk 过滤掉静音和噪音之后的纯语音部分。
            state: wave_chunk 的状态值。各取值含义如下。
                0：整个 wave_chunk 都是静音，或者只有中间有很短的语音
                1：wave_chunk 从静音变成了语音，也就是 wave_chunk 开始是静音，结尾是语音。
                2：wave_chunk 的开始和结尾都是语音。
                3：wave_chunk 从语音变成了静音，也就是开头是语音，结尾是静音。
        """

        offset = np.all(post_label==0)  #all frame is zero, this chunk should be treat as one offset
        onset = np.all(post_label==1)   #all frame is one, this chunk should be treat as one onset

        if not offset and not onset: # frames have both 0 and 1, we do some more smoothing, and detect onset/offset
            '''fill 1 to short valley'''
            for i in range(post_label.shape[0]):
                if i < post_label.shape[0] - 1:
                    if post_label[i] == 1 and post_label[
                            i + 1] == 0:  # offset detection
                        offset = True
                        offset_point = i
                    if post_label[i] == 0 and post_label[
                            i + 1] == 1 and offset:  # offset -> onset detection
                        if i - offset_point < on_off_length:
                            post_label[offset_point:i + 1] = 1  # fill 1 to valley
                            offset = False
            '''remove impulse like detection: change 1 to 0'''
            for i in range(post_label.shape[0]):
                if i < post_label.shape[0] - 1:
                    if post_label[i] == 0 and post_label[
                            i + 1] == 1:  # onset detection
                        onset = True
                        onset_point = i
                    if post_label[i] == 1 and post_label[
                            i + 1] == 0 and onset:  # onset -> offset detection
                        if i - onset_point < off_on_length:
                            post_label[onset_point:i + 1] = 0  # fill 0 to hill
                            onset = False
            '''hang before & over: expand the span of speech frames'''
            for i in range(post_label.shape[0]):
                if i < post_label.shape[0] - 1:
                    if post_label[i] == 0 and post_label[
                            i + 1] == 1:  # onset detection
                        onset = True
                        if i - hang_before < 0:
                            post_label[0:i + 1] = 1
                        else:
                            post_label[i - hang_before:i + 1] = 1
                    if post_label[i] == 1 and post_label[
                            i + 1] == 0 and onset:  # onset -> offset detection
                        onset = False
                        #print(i)
                        if i + hang_over > post_label.shape[0]:
                            post_label[i:] = 1
                        else:
                            post_label[i:i + hang_over] = 1

        if self.state == 0:
            if onset:
                self.state = 1
        elif self.state == 1:
            if offset:
                self.state = 3
            else:
                self.state = 2
        elif self.state == 2:
            if offset:
                self.state = 3
        elif self.state == 3:
            if onset:
                self.state = 1
            else:
                self.state = 0

        return self.state, post_label

    def vad_endpoint(self, post_label):
        """
        根据输入的预测标签，结合当前chunk状态进行断句
        :param post_label:  当前chunk输出的标签
        :return: vad_state: 用于ASR的VAD_state:
                    0: 表示当前chunk无需断句，对应ASR的中间结果
                    1: 表示当前累积的静音长度超过限值，需要断句
                    2: 表示当前拖尾静音长度超过限值，需要断句
                    3: 表示当前累积的所有音频长度超过限值，需要断句
        """
        self.endpoint_state = 0
        # 累积总的帧数
        self.total_frames+=len(post_label)

        # 当前chunk全为0，或者从1到0
        if self.state == 0 or self.state == 3:
            for i in range(len(post_label)):
                if post_label[len(post_label)-1-i]==0:
                    self.sil_frames += 1
                else:
                    break
            if not self.has_speech:
                if self.sil_frames*self.resolution >= self.max_silence_length:
                    self.endpoint_state = 1
                    return 1
            else:
                if self.sil_frames*self.resolution*1000 >= self.max_trailing_silence:
                    self.endpoint_state = 2
                    return 2

        # 当前chunk从0到1， 或者全为1
        elif self.state == 1 or self.state == 2:
            self.has_speech = True
            self.sil_frames = 0

        if self.total_frames * self.resolution >= self.max_speech_length:
            self.endpoint_state = 3
            return 3

        return self.endpoint_state


    def get_speech_time_stamp(self, prediction, min_speech_frames=20, min_sil_frames=20):
        '''
        用于获取当前chunk内部的语音片段，即当前输入音频帧中是语音的采样点范围
        :param prediction: vad prediction labels
        :param min_speech_frames: min speech duration in frames
        :param min_sil_frames: min sil frames between two speech segment,
                    two speech segments will be merged if sil between is less than this value

        :return: time stamp in second, 语音段时间戳，作为整段计算时，可以直接减去时延帧数，得到更准确的时间戳
        '''
        # 将连续相同标签进行合并，得到时间戳
        time_stamp = decode_with_timestamps(prediction)
        # 限制语音段长度超过最小长度, 并且人为将流式推断时的时延减去
        speech_time_stamp = [
            [max((stamp[0]-self.latency_frames),0),
            max((stamp[1]-self.latency_frames),0)] for stamp in time_stamp
            if (stamp[1] - stamp[0]) > min_speech_frames
        ]
        merged_time_stamp = []
        for time_stamp in speech_time_stamp:
            if len(merged_time_stamp) == 0:
                merged_time_stamp.append(time_stamp)
                continue
            # merge time stamp
            if (time_stamp[0] - merged_time_stamp[-1][1]) < min_sil_frames:
                merged_time_stamp[-1][1] = time_stamp[1]
            else :
                merged_time_stamp.append(time_stamp)

        return np.array(merged_time_stamp)  * self.resolution   # in second
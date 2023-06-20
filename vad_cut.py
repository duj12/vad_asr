import torch
import torch.nn as nn
import numpy as np
import librosa

def crnn(inputdim=64, outputdim=2, pretrained_from=None, **kwargs):
    model = CRNN(inputdim, outputdim, **kwargs)
    if pretrained_from:
        state = torch.load(pretrained_from, map_location='cpu')
        model.load_state_dict(state, strict=False)
    return model


class LinearSoftPool(nn.Module):
    """LinearSoftPool

    Linear softmax, takes logits and returns a probability, near to the actual maximum value.
    Taken from the paper:

        A Comparison of Five Multiple Instance Learning Pooling Functions for Sound Event Detection with Weak Labeling
    https://arxiv.org/abs/1810.09050

    """
    def __init__(self, pooldim=1):
        super().__init__()
        self.pooldim = pooldim

    def forward(self, logits, time_decision):
        return (time_decision**2).sum(self.pooldim) / time_decision.sum(
            self.pooldim)


class MeanPool(nn.Module):
    def __init__(self, pooldim=1):
        super().__init__()
        self.pooldim = pooldim

    def forward(self, logits, decision):
        return torch.mean(decision, dim=self.pooldim)


class AttentionPool(nn.Module):
    """docstring for AttentionPool"""
    def __init__(self, inputdim, outputdim=10, pooldim=1, **kwargs):
        super().__init__()
        self.inputdim = inputdim
        self.outputdim = outputdim
        self.pooldim = pooldim
        self.transform = nn.Linear(inputdim, outputdim)
        self.activ = nn.Softmax(dim=self.pooldim)
        self.eps = 1e-7

    def forward(self, logits, decision):
        # Input is (B, T, D)
        # B, T , D
        w = self.activ(self.transform(logits))
        detect = (decision * w).sum(
            self.pooldim) / (w.sum(self.pooldim) + self.eps)
        # B, T, D
        return detect


def parse_poolingfunction(poolingfunction_name='mean', **kwargs):
    """parse_poolingfunction
    A heler function to parse any temporal pooling
    Pooling is done on dimension 1

    :param poolingfunction_name:
    :param **kwargs:
    """
    poolingfunction_name = poolingfunction_name.lower()
    if poolingfunction_name == 'mean':
        return MeanPool(pooldim=1)
    elif poolingfunction_name == 'linear':
        return LinearSoftPool(pooldim=1)
    elif poolingfunction_name == 'attention':
        return AttentionPool(inputdim=kwargs['inputdim'],
                             outputdim=kwargs['outputdim'])


class Block2D(nn.Module):
    def __init__(self, cin, cout, kernel_size=3, padding=1):
        super().__init__()
        self.block = nn.Sequential(
            nn.BatchNorm2d(cin),
            nn.Conv2d(cin,
                      cout,
                      kernel_size=kernel_size,
                      padding=padding,
                      bias=False),
            nn.LeakyReLU(inplace=True, negative_slope=0.1))

    def forward(self, x):
        return self.block(x)


def init_weights(m):
    if isinstance(m, (nn.Conv2d, nn.Conv1d)):
        nn.init.kaiming_normal_(m.weight)
        if m.bias is not None:
            nn.init.constant_(m.bias, 0)
    elif isinstance(m, nn.BatchNorm2d):
        nn.init.constant_(m.weight, 1)
        if m.bias is not None:
            nn.init.constant_(m.bias, 0)
    if isinstance(m, nn.Linear):
        nn.init.kaiming_uniform_(m.weight)
        if m.bias is not None:
            nn.init.constant_(m.bias, 0)


class CRNN(nn.Module):
    def __init__(self, inputdim, outputdim, **kwargs):
        super().__init__()
        self.features = nn.Sequential(
            Block2D(1, 32),
            nn.LPPool2d(4, (2, 4)),
            Block2D(32, 128),
            Block2D(128, 128),
            nn.LPPool2d(4, (2, 4)),
            Block2D(128, 128),
            Block2D(128, 128),
            nn.LPPool2d(4, (1, 4)),
            nn.Dropout(0.3),
        )
        with torch.no_grad():
            rnn_input_dim = self.features(torch.randn(1, 1, 500, inputdim)).shape
            rnn_input_dim = rnn_input_dim[1] * rnn_input_dim[-1]
        self.gru = nn.GRU(rnn_input_dim, 128, bidirectional=kwargs.get('gru_bidirection', True), batch_first=True)
        self.temp_pool = parse_poolingfunction(kwargs.get('temppool', 'linear'),
                                               inputdim=256 if kwargs.get('gru_bidirection', True) else 128,
                                               outputdim=outputdim)
        self.outputlayer = nn.Linear(256 if kwargs.get('gru_bidirection', True) else 128, outputdim)
        self.features.apply(init_weights)
        self.outputlayer.apply(init_weights)

    def forward(self, x, upsample=True):
        batch, time, dim = x.shape
        x = x.unsqueeze(1)
        x = self.features(x)
        x = x.transpose(1, 2).contiguous().flatten(-2)
        x, _ = self.gru(x)
        decision_time = torch.sigmoid(self.outputlayer(x)).clamp(1e-7, 1.)
        if upsample:
            decision_time = torch.nn.functional.interpolate(
                decision_time.transpose(1, 2),
                time,
                mode='linear',
                align_corners=False).transpose(1, 2)
        decision = self.temp_pool(x, decision_time).clamp(1e-7, 1.).squeeze(1)
        return decision, decision_time

    def forward_stream_vad(self, x, h, upsample=True):
        batch, time, dim = x.shape
        x = x.unsqueeze(1)
        x = self.features(x)
        x = x.transpose(1, 2).contiguous().flatten(-2)
        x, h = self.gru(x, h)
        vad_post = torch.sigmoid(self.outputlayer(x)).clamp(1e-7, 1.)
        if upsample:
            vad_post = torch.nn.functional.interpolate(
                vad_post.transpose(1, 2),
                time,
                mode='linear',
                align_corners=False).transpose(1, 2)
        return vad_post, h

    def forward_stream_vad_cnn(self, x):
        x = x.unsqueeze(1)
        x = self.features(x)
        x = x.transpose(1, 2).contiguous().flatten(-2)
        return x
    def forward_stream_vad_gru(self, x, h):
        x, h = self.gru(x, h)
        vad_post = torch.sigmoid(self.outputlayer(x)).clamp(1e-7, 1.)
        return vad_post, h
    def forward_stream_vad_upsample(self, vad_post, upsample_time_dimension):
        vad_post = torch.nn.functional.interpolate(
            vad_post.transpose(1, 2),
            upsample_time_dimension,
            mode='linear',
            align_corners=False).transpose(1, 2)
        return vad_post


class CRNN_VAD_STREAM(torch.nn.Module):
    '''
    Online VAD module, with 5-CNN + GRU.
    Because of the existing of 4-time down-sampling in CNN,
    the frame context is best to be 4*N.
    The total reception field=11, left and right frames should be larger than 11
    chunk_length(sample counts) = (total_frame + 10) * 200
    '''
    def __init__(self, model_path,
                 left_frames=12, central_frames=36, right_frames=12,
                 output_dim=2, gru_bidirection=False, device='cpu'):
        super().__init__()
        self.model = crnn(pretrained_from=model_path,
                          outputdim=output_dim,
                          gru_bidirection=gru_bidirection).to(device)
        self.resolution = 0.0125   #帧率为80
        self.speech_label_idx = np.array(1)   # 0: None-Speech, 1: Speech
        self.sample_rate = 16000
        self.mel_nfft = 2048
        self.mel_win_length = 400  #对应16khz音频
        self.mel_hop_length = 200  #对应16khz音频，帧率为80
        self.mel_bin = 64
        self.EPS = np.spacing(1)
        self.device = device
        self.model.eval()
        self.cnn_subsample_factor = 4

        self.left_frames = left_frames #at least 12, best to be 16
        self.central_frames = central_frames
        self.right_frames = right_frames #at least 12, best to be 16

        self.N_frames = self.left_frames+self.central_frames+self.right_frames
        self.left_context = self.left_frames * self.mel_hop_length
        self.central_context = self.central_frames * self.mel_hop_length
        self.right_context = self.right_frames * self.mel_hop_length
        #chunk_length must bigger than (N_frasmes-1)*mel_hop_length+mel_nfft to get full frames
        #We directly make chunk_length = (N_frames + 10) * hop_length, since mel_nfft~10*hop_length
        # 16000对应(16,38,16)/(12,46,12); 18000对应(16,48,16)/(12,64,12)
        self.nfft2hoplen = self.mel_nfft // self.mel_hop_length  # 10
        self.chunk_length = (self.N_frames+self.nfft2hoplen) * self.mel_hop_length

        self.gru_vector = None

        # vad_judgement related
        self.max_speech_length = -1 #限制最长语音段长度(For ASR task).
        self.state = 0 #0:SIL, 1:SPEECH_START, 2:SPEECH_ON, 3:SPEECH_END
        self.sil_frames = 0   #进入SPEECH_END或SIL状态之后已经累计的静音帧
        self.speech_frames = 0  #进入SPEECH＿START或SPEECH_ON状态之后已经累计的语音帧

    def clear_gru_buffer(self):
        self.gru_vector = None

    def forward(self, wave, threshold=(0.5, 0.2), min_speech_frames=10):
        '''
        :param wave: chunked wave samples.
        :param threshold:  a tuple, with one or two elements. Default in double threshold.
        :param min_speech_frames: min speech duration in frames
        :return: vad probability,  speech sample stamp, thresholded vad prediction
        '''
        logmel_feat = self.extract_mel_online(wave)
        with torch.no_grad():
            feature = torch.as_tensor(logmel_feat).to(self.device)
            feature = feature.unsqueeze(0)  # add one dimension to match forward
            _, time, _ = feature.shape
            output_frame_num = time - (self.left_context + self.right_context) // self.mel_hop_length
            cnn_feature = self.model.forward_stream_vad_cnn(feature)
            down_sample_factor = self.mel_hop_length * self.cnn_subsample_factor
            # since we only output central, we should keep the context vector at position of central
            cnn_feature_to_gru = cnn_feature[:, self.left_context // down_sample_factor:
                                                - self.right_context // down_sample_factor, :]
            gru_output, self.gru_vector = self.model.forward_stream_vad_gru(cnn_feature_to_gru, self.gru_vector)
            vad_chunk = self.model.forward_stream_vad_upsample(gru_output, output_frame_num)
            vad_post = vad_chunk[0].cpu().numpy()  # single-batch
        time_stamp, hard_pred = self.post_process(vad_post, threshold=threshold, min_speech_frames=min_speech_frames)
        return vad_post, time_stamp, hard_pred

    def extract_mel_online(self, wave_chunk):
        """
        because mel_nfft=2048, in each chunk, when extract_mel_feat, we remain 2000 samples in each process.
        :param wave_chunk: the input wave.
        :return:
        """
        log_mel = np.log(
            librosa.feature.melspectrogram(wave_chunk.astype(np.float32), self.sample_rate,
                                           n_fft=self.mel_nfft, win_length=self.mel_win_length,
                                           hop_length=self.mel_hop_length, n_mels=self.mel_bin,
                                           center=False) + self.EPS).T
        return log_mel

    def post_process(self, vad_post, threshold=(0.5, 0.2), min_speech_frames=10):
        '''
        :param vad_post: vad probability
        :param threshold: a tuple, with one or two elements. Default in double threshold.
        :param min_speech_frames: min speech duration in frames
        :return: time stamp in sample count, thresholded prediction of vad
        '''
        if len(threshold) == 1:
            postprocessing_method = binarize  # 单阈值为(0.5,)，不进行任何平滑
        else:
            postprocessing_method = double_threshold  # 双阈值默认为(0.5, 0.2)进行平滑

        thresholded_prediction = postprocessing_method(vad_post, *threshold)
        # 将连续相同标签进行合并，得到时间戳，三个元素分别为：语音/非语音，起始时间，结束时间
        time_stamp = decode_with_timestamps(thresholded_prediction)
        # 选出语音时间戳(0:非语音，1：语音)，限制语音段长度超过最小长度
        speech_time_stamp = [[stamp[1], stamp[2]] for stamp in time_stamp
                             if(stamp[0]==1 ) and (stamp[2]-stamp[1])>min_speech_frames]
        speech_sample_stamp = np.array(speech_time_stamp)*self.resolution*self.sample_rate #将帧号转换为采样点数
        return speech_sample_stamp.astype(int), thresholded_prediction.astype(int)

    def vad_judgement(self, wave_chunk, post_label,
                      off_on_length=20, on_off_length=20,
                      hang_before=10, hang_over=10):
        """
        Given input wave_chunk, output the real speech in wave according to the VAD-label.
        And return the state of current chunk. The state could be the following 4 types:
        SIL, SPEECH_START, SPEECH_ON, SPEECH_END.
        :param wave_chunk: the real wave without any left and right context.
        :param post_label: the label that VAD model give, in frame.
        :param off_on_length: from sil to speech, we need how many speech frames.
        :param on_off_length: from speech to sil, we need how many sil frames.
        :param hang_before: if speech onset is detection, we add how many speech frames before the onset point.
        :param hang_over: if speech offset is detection, we add how many speech frames after the offset point.
        :return:
            output_speech: the speech in the input wave_chunk.
            state: the speech or silence state at current chunk.
        """
        #offset = False  # from 1 to 0
        #onset = False  # from 0 to 1
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

        change_indices = find_contiguous_regions(post_label)
        output_speech = []
        for row in change_indices:
            speech_seg = wave_chunk[row[0]*self.mel_hop_length: row[1]*self.mel_hop_length]
            output_speech += speech_seg.tolist()

        return np.array(output_speech), self.state

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
    import matplotlib.pyplot as plt
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
    import sklearn.metrics as skmetrics
    assert (len(reference) == len(hypothesis))
    def roc(y_true, y_pred, average=None):
        return skmetrics.roc_auc_score(y_true, y_pred, average=average)
    def precision_recall_fscore_support(y_true, y_pred, average=None):
        return skmetrics.precision_recall_fscore_support(y_true,
                                                         y_pred,
                                                         average=average)
    def confusion_matrix(y_true, y_pred):
        return skmetrics.confusion_matrix(y_true, y_pred)
    def obtain_error_rates(y_true, y_pred, threshold=0.5):
        negatives = y_pred[np.where(y_true == 0)]
        positives = y_pred[np.where(y_true == 1)]
        Pfa = np.sum(negatives >= threshold) / negatives.size
        Pmiss = np.sum(positives < threshold) / positives.size
        return Pfa, Pmiss

    speech_frame_ground_truth = np.concatenate(reference, axis=0)
    speech_frame_predictions = np.concatenate(prediction, axis=0)
    speech_frame_prob_predictions = np.concatenate(hypothesis, axis=0)
    vad_results = []
    tn, fp, fn, tp = confusion_matrix(
        speech_frame_ground_truth, speech_frame_predictions).ravel()
    fer = 100 * ((fp + fn) / len(speech_frame_ground_truth))
    acc = 100 * ((tp + tn) / (len(speech_frame_ground_truth)))
    p_miss = 100 * (fn / (fn + tp))
    p_fa = 100 * (fp / (fp + tn))
    for i in [0.1, 0.3, 0.5, 0.7, 0.9]:
        mp_fa, mp_miss = obtain_error_rates(
            speech_frame_ground_truth, speech_frame_prob_predictions, i)
        tn, fp, fn, tp = confusion_matrix(
            speech_frame_ground_truth,
            speech_frame_prob_predictions > i).ravel()
        sub_fer = 100 * ((fp + fn) / len(speech_frame_ground_truth))
        print(
            f"PFa {100 * mp_fa:.2f} Pmiss {100 * mp_miss:.2f} FER {sub_fer:.2f} t: {i:.2f}"
        )

    auc = roc(speech_frame_ground_truth, speech_frame_prob_predictions) * 100
    for avgtype in ('micro', 'macro', 'binary'):
        precision, recall, f1, _ = precision_recall_fscore_support(
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


def decode_with_timestamps(labels):
    """decode_with_timestamps
    Decodes the predicted label array (2d) into a list of
    [(Labelname, onset, offset), ...]

    :param encoder: Encoder during training
    :type encoder: pre.MultiLabelBinarizer
    :param labels: n-dim array
    :type labels: np.array
    """
    if labels.ndim == 3:
        return [_decode_with_timestamps(lab) for lab in labels]
    else:
        return _decode_with_timestamps(labels)


def _decode_with_timestamps(labels):
    result_labels = []
    for i, label_column in enumerate(labels.T):
        change_indices = find_contiguous_regions(label_column)
        # append [onset, offset] in the result list
        for row in change_indices:
            result_labels.append((i, row[0], row[1]))
    return result_labels


def sma_filter(x, window_size, axis=1):
    """sma_filter

    :param x: Input numpy array,
    :param window_size: filter size
    :param axis: over which axis ( usually time ) to apply
    """
    # 1 is time axis
    kernel = np.ones((window_size, )) / window_size

    def moving_average(arr):
        return np.convolve(arr, kernel, 'same')

    return np.apply_along_axis(moving_average, axis, x)


def median_filter(x, window_size, threshold=0.5):
    """median_filter

    :param x: input prediction array of shape (B, T, C) or (B, T).
        Input is a sequence of probabilities 0 <= x <= 1
    :param window_size: An integer to use
    :param threshold: Binary thresholding threshold
    """
    x = binarize(x, threshold=threshold)
    if x.ndim == 3:
        size = (1, window_size, 1)
    elif x.ndim == 2 and x.shape[0] == 1:
        # Assume input is class-specific median filtering
        # E.g, Batch x Time  [1, 501]
        size = (1, window_size)
    elif x.ndim == 2 and x.shape[0] > 1:
        # Assume input is standard median pooling, class-independent
        # E.g., Time x Class [501, 10]
        size = (window_size, 1)
    import scipy
    return scipy.ndimage.median_filter(x, size=size)


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


def vad_demo():
    import soundfile as sf
    import os
    from loguru import logger

    SAMPLE_RATE = 16000
    DEVICE = 'cpu'  # cpu is fast enough.
    if torch.cuda.is_available():
        DEVICE = 'cuda'
    DEVICE = torch.device(DEVICE)

    # #测试vad性能
    # threshold = (0.5,)   #(0.5,), (0.5, 0.2)
    # save_cutted_wav = False
    # wav_list = '/data/megastore/Projects/DuJing/code/Datadriven-VAD/data/test/test.list'
    # ref_list = '/data/megastore/Projects/DuJing/code/Datadriven-VAD/data/test/test.vad'
    # output_path = '/data/megastore/Projects/DuJing/code/Datadriven-VAD/data/test/XmovVAD_stream/test'

    #实际切分音频
    threshold = (0.5, 0.2)
    save_cutted_wav = True
    wav_list = 'wav.list'
    ref_list = None
    output_path="output_stream"
    output_path = '/data/megastore/Projects/DuJing/deployment/crnn-vad0/output_stream'

    f_wav = open(wav_list, 'r')
    if ref_list:
        f_ref = open(ref_list, 'r')
    hypothesis = []
    reference = []
    prediction = []

    def process_one_thread(file_list, output_path):
        # vad_ckpt_path =  'pretrained_models/xmov/hard.pt'  #离线模型
        vad_ckpt_path = 'pretrained_models/xmov/stream-hard.pt'  # 流式模型

        vad_model = CRNN_VAD_STREAM(vad_ckpt_path, device=DEVICE)  # each thread with one model
        for line in file_list:
            wav_path = line.strip()
            logger.info(f'processing {wav_path}')

            if output_path:
                from pathlib import Path
                output_path = Path(output_path)
                output_path.mkdir(parents=True, exist_ok=True)
            wav_name = wav_path.split('/')[-1].split('.')[0]

            #TODO: read audio in stream fashion
            _, file_extension = os.path.splitext(wav_path)
            if file_extension in ['.mp3', '.wav', '.m4a', '.aac']:
                wave, sr = librosa.load(wav_path)
            else:
                raise NotImplementedError(f'Audio extension {file_extension} not supported... yet ;)')
            if wave.ndim > 1:
                wave = wave.mean(-1)
            if sr != 16000:
                wave = librosa.resample(wave, sr, target_sr=16000)

            #clear GRU buffer at the begining of each wave processing
            vad_model.clear_gru_buffer()
            vad_post = np.empty((0, 2), dtype=float)
            hard_pred = np.empty((0, 2), dtype=int)
            speech_segment = []
            speech_segment_num = 0
            wav_len = wave.shape[-1]

            wave_padded = wave
            # padding in the begining, for better performence in the begining
            wave_padded = np.pad(wave, (vad_model.mel_nfft//2), mode='reflect')
            # padding with left and right context. 提前给音频pad上前后文，实际上不会输出这部分，只会输出中间部分
            wave_padded = np.pad(wave_padded, (vad_model.left_context, vad_model.right_context) )

            wav_padded_len = wave_padded.shape[-1]
            for chunk_id, start in enumerate(range(0, wav_padded_len, vad_model.central_context)):
                end = min(start + vad_model.chunk_length, wav_padded_len)
                wave_chunk = wave_padded[start:end] #输入wav_chunk为包含了前后文的
                # wave left is less than chunk_length or a little bit more than chunk_length
                if wav_padded_len - start < vad_model.chunk_length + vad_model.right_context + vad_model.mel_nfft//2:
                    wave_chunk = wave_padded[start:]
                    start = wav_padded_len # to break the loop

                #模型前向计算，输出的时间戳对应的是输入采样点经过[left:left+central]切片之后的语音片段
                vad_post_chunk, time_stamp_chunk, hard_pred_chunk = vad_model(wave_chunk, threshold)
                speech_label = hard_pred_chunk[:,1]

                #输出VAD判决之后保留的语音，以及当前chunk所处状态：SIL, SPEECH_START, SPEECH_ON, SPEECH_END
                #SIL 表明当前chunk起始和结束段都是静音，SPEECH_END 表明当前chunk从语音转为静音
                #SPEECH_ON 表明当前chunk起始和结束段都是语音，SPEECH_START 表明当前chunk从静音转为语音
                output_speech_chunk, chunk_state = vad_model.vad_judgement(
                    wave_chunk[vad_model.left_context: vad_model.left_context+vad_model.central_context], speech_label)

                # 根据当前状态累积输出音频，符合条件时直接保存音频片段进行check
                #speech_segment += output_speech_chunk.tolist()  #只保存过滤掉静音之后的音频，会导致音频比较
                if chunk_state!=0 :  #只要不是整段静音就拼起来
                    speech_segment += wave_chunk[vad_model.left_context: vad_model.left_context+vad_model.central_context].tolist()
                if chunk_state == 3  or len(speech_segment) >= 15*16000: #SPEECH_END or reach 15 second
                    # 只要一段音频结束，就输出切分之后的wav，以便检查和分析
                    if output_path and save_cutted_wav:
                        sf.write(os.path.join(output_path, '{}_{:05d}.wav'.format(wav_name, speech_segment_num)),
                           np.array(speech_segment), samplerate=SAMPLE_RATE)
                    speech_segment_num += 1
                    speech_segment = []  # clear the segment

                vad_post = np.append(vad_post, vad_post_chunk, axis=0)
                hard_pred = np.append(hard_pred, hard_pred_chunk, axis=0)

                if start == wav_padded_len:
                    break

            # The following code is for algorithm validation
            #cutted_wave = vad_model.cut_wav_in_batch(wave, time_stamp)  #in some case we need cutted wav in batch
            #zero-th colume is Non-Speech, 1st column is Speech
            speech_prob = vad_post[:, 1]
            pred_label = hard_pred[:, 1]
            hypothesis.append(speech_prob)
            prediction.append(pred_label)

            # if save_cutted_wav:
            #     #这里的时间戳是整条音频的时间戳，并非根据每个chunk实时得到的
            #     time_stamp, _ = vad_model.post_process(vad_post, threshold)
            #     for i, sample_stamp in enumerate(time_stamp):
            #         # 输出切分之后的wav，以便检查和分析
            #         if output_path:
            #             sf.write(os.path.join(output_path, '{}_{:05d}.wav'.format(wav_name, i)),
            #                wave[min(sample_stamp[0],wav_len): min(sample_stamp[1], wav_len)], samplerate=SAMPLE_RATE)

            if ref_list:
                # 提供了参考标注，以及预测标签
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
            # # # 画出模型输出后验的结果和语音波形
            # else:
            #     plot_wav_and_vad(wave, None, speech_prob, f"{output_path}/{wav_name}.png")


    file_list=[]
    for i, line in enumerate(f_wav.readlines()):
        file_list.append(line)

    th_cnt = min(1, len(file_list))

    if th_cnt==1:
        process_one_thread(file_list, output_path)
    else:
        import threading
        total_cnt = len(file_list)
        job_per_th = total_cnt // th_cnt if(total_cnt%th_cnt==0) else total_cnt // th_cnt + 1
        threads = []
        for th in range(th_cnt):
            start_idx = th * job_per_th
            end_idx = min((th+1) * job_per_th, total_cnt)
            threads.append(threading.Thread(target=process_one_thread, args=(file_list[start_idx:end_idx], output_path)))
        for th in range(th_cnt):
            threads[th].start()

    if ref_list:
        logger.info("Calculating VAD measures ... ")
        measure_vad_performance(reference, prediction, hypothesis)
    f_wav.close()
    if ref_list:
        f_ref.close()


if __name__ == "__main__":
    vad_demo()
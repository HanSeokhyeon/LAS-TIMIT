import numpy as np
import yaml
from spikegram.gammatone import get_gammatone_filter
config = yaml.load(open("config/spikegram_config.yaml", 'r'), Loader=yaml.FullLoader)


def preprocess_spikegram():
    data_list = []
    data_list.append(np.loadtxt(config['data_path']['train'], delimiter=',', dtype=np.str))
    data_list.append(np.loadtxt(config['data_path']['valid'], delimiter=',', dtype=np.str))
    data_list.append(np.loadtxt(config['data_path']['test'], delimiter=',', dtype=np.str))

    global gammatone_filter
    gammatone_filter = get_gammatone_filter()
    for now_list in data_list:
        for filename in now_list:
            make_spikegram(filename)


def make_spikegram(filename):
    frame = config['common']['frame']
    twin = config['common']['twin']
    number_of_channel = config['common']['number_of_channel']


    signal = np.fromfile("dataset/TIMIT/{}.WAV".format(filename), dtype=np.int16)[512:]
    number_of_frame = signal.shape[0] // frame + 1

    signal = np.pad(signal, pad_width=(0, number_of_frame*frame - signal.shape[0] + twin))
    result = []
    psnr = 0
    for nof in range(number_of_frame):
        now_signal = signal[nof*frame:(nof+1)*frame + twin].astype(np.float64)
        original_signal = signal[nof*frame:(nof+1)*frame]
        restored_signal = np.zeros(frame+twin)

        result_frame = []

        point = 1
        while psnr < 50:
            corr = np.zeros((number_of_channel, frame))
            for i in range(number_of_channel):
                corr[i] = np.correlate(now_signal, gammatone_filter[i])[:-1]
            max_x, max_y = np.unravel_index(np.argmax(np.abs(corr)), corr.shape)

            now_signal[max_y:max_y+twin] -= corr[max_x, max_y] * gammatone_filter[max_x]
            restored_signal[max_y:max_y+twin] += corr[max_x, max_y] * gammatone_filter[max_x]

            psnr = calculate_psnr(original_signal, restored_signal, frame)

            result_frame.append([max_x, corr[max_x, max_y], max_y, psnr])

            print("\r{}\t{}\t{},".format(point, psnr, corr[max_x, max_y]))
            point += 1


def calculate_psnr(original, restored, size):
    mse = np.sum(np.power(original-restored[:size], 2))
    max_i = np.power(32767*2, 2)*size

    return 10 * np.log10(max_i / mse)

if __name__ == '__main__':
    preprocess_spikegram()

import numpy as np
import yaml
from spikegram.gammatone import get_gammatone_filter
config = yaml.load(open("config/spikegram_config.yaml", 'r'), Loader=yaml.FullLoader)


def preprocess_spikegram():
    data_list = []
    data_list.append(np.loadtxt(config['data_path']['train'], delimiter=',', dtype=np.str))
    data_list.append(np.loadtxt(config['data_path']['valid'], delimiter=',', dtype=np.str))
    data_list.append(np.loadtxt(config['data_path']['test'], delimiter=',', dtype=np.str))

    for now_list in data_list:
        for filename in now_list:
            make_spikegram(filename)


def make_spikegram(filename):
    frame = config['common']['frame']
    twin = config['common']['twin']
    number_of_channel = config['common']['number_of_channel']

    gammatone_filter = get_gammatone_filter()

    signal = np.fromfile("dataset/TIMIT/{}.WAV".format(filename), dtype=np.int16)[512:]
    number_of_frame = signal.shape[0] // frame + 1

    signal = np.pad(signal, pad_width=(0, number_of_frame*frame - signal.shape[0] + twin))

    for nof in range(number_of_frame):


        now_signal = signal[nof*frame:(nof+1)*frame + twin]
        corr = np.zeros((number_of_channel, frame))
        for i in range(number_of_channel):
            corr[i] = np.correlate(now_signal, gammatone_filter[i])[:-1]
        max_x, max_y = np.unravel_index(np.argmax(np.abs(corr)), corr.shape)

        pass



if __name__ == '__main__':
    preprocess_spikegram()

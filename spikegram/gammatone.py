import numpy as np
import yaml
import matplotlib.pyplot as plt

config = yaml.load(open("gammatone_config.yaml", 'r'), Loader=yaml.FullLoader)


def get_gammatone_filter():
    number_of_channel = config['gammatone']['number_of_channel']
    twin = config['gammatone']['twin']
    sample_rate = config['common']['sample_rate']

    center_freq, erb_filter = make_erb_filter()

    bandwidth = 1.019 * erb_filter

    t = 1 / sample_rate

    gammatone_filter = np.zeros((number_of_channel, twin))
    for i in range(number_of_channel):
        for j in range(twin):
            gammatone_filter[i][j] = np.power(j*t, 3) * np.exp(-2 * np.pi * bandwidth[i] * t * j) * np.cos(2 * np.pi * center_freq[i] * t * j)

    filter_sum = np.sum(np.power(gammatone_filter, 2), axis=1)
    gammatone_filter = gammatone_filter / np.power(filter_sum, 0.5)[:, None]

    # plt.plot(gammatone_filter[2].T)
    # plt.show()

    return gammatone_filter


def make_erb_filter():
    fmax = config['common']['sample_rate'] // 2
    min_freq = config['gammatone']['min_freq']
    number_of_channel = config['gammatone']['number_of_channel']
    ear_q = config['gammatone']['ear_q']
    min_bandwidth = config['gammatone']['min_bandwidth']
    order = config['gammatone']['order']

    center_freq = np.zeros(number_of_channel)
    center_freq[0] = config['gammatone']['min_freq']
    for i in range(1, number_of_channel+1):
        center_freq[number_of_channel - i] = -(ear_q*min_bandwidth) \
                                             + np.exp(i * (-np.log(fmax + ear_q * min_bandwidth)
                                             + np.log(min_freq + ear_q * min_bandwidth)) / number_of_channel) \
                                             * (fmax + ear_q * min_bandwidth)

    erb = np.zeros(number_of_channel)
    for i in range(number_of_channel):
        erb[i] = np.power(np.power(center_freq[i]/ear_q, order) + np.power(min_bandwidth, order), 1/order)

    return center_freq, erb


if __name__ == '__main__':
    get_gammatone_filter()

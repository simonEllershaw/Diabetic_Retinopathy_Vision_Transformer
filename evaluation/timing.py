import numpy as np
import torch

def calc_image_throughput(model, device, img_size, batch_size):
    # https://towardsdatascience.com/the-correct-way-to-measure-inference-time-of-deep-neural-networks-304a54e5187f
    dummy_input = torch.randn(batch_size, 3, img_size, img_size, dtype=torch.float).to(device)
    repetitions=100
    time_log = np.zeros(repetitions)
    with torch.no_grad():
        for rep in range(repetitions):
            starter, ender = torch.cuda.Event(enable_timing=True), torch.cuda.Event(enable_timing=True)
            starter.record()
            _ = model(dummy_input)
            ender.record()
            torch.cuda.synchronize()
            curr_time = starter.elapsed_time(ender)/1000
            time_log[rep] = curr_time
    throughput = batch_size/time_log
    throughput_mean = np.mean(throughput)
    throughput_std = np.std(throughput)
    return throughput_mean, throughput_std
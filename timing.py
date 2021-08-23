import os
import time
import pickle
import sys
import numpy as np
import torch
import evaluate
#https://towardsdatascience.com/the-correct-way-to-measure-inference-time-of-deep-neural-networks-304a54e5187f
if __name__ == "__main__":
    # Set up directory for experiment
    print(sys.argv)
    model_name = sys.argv[1] if len(sys.argv) > 1 else "vit_small_patch16_224_in21k"
    img_size = int(sys.argv[2]) if len(sys.argv) > 2 else 384
    batch_size = int(sys.argv[3]) if len(sys.argv) > 3 else 16
    resize_model = True if len(sys.argv) > 4 and int(sys.argv[4]) > 0 else False
    print(model_name, img_size, batch_size, resize_model)
    
    device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")

    
    if resize_model:
        model = evaluate.load_model(model_name, device, ["A", "B"], model_resize=img_size)
    else:
        model = evaluate.load_model(model_name, device, ["A", "B"])

    dummy_input = torch.randn(batch_size, 3, img_size, img_size, dtype=torch.float).to(device)
    dummy_label = torch.randint(1, (batch_size, 3, img_size, img_size))
    repetitions=1000
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
    throughput = batch_size/total_time
    throughput_mean = np.mean(throughput)
    throughput_std = np.std(throughput)
    print(f"Final Throughput:{throughput_mean}, {throughput_std}")

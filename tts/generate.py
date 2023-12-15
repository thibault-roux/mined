from transformers import VitsModel, AutoTokenizer
import torch

model = VitsModel.from_pretrained("facebook/mms-tts-fra")
tokenizer = AutoTokenizer.from_pretrained("facebook/mms-tts-fra")

text = "salut tu vas bien"
inputs = tokenizer(text, return_tensors="pt")

with torch.no_grad():
    output = model(**inputs).waveform

import scipy

import numpy as np

# Convert the PyTorch tensor to a NumPy array
output_np = output.numpy()

# Scale the values to the expected range for 16-bit PCM audio (usually between -32768 and 32767)
output_np_scaled = np.int16(output_np * 32767)

# Write the WAV file using the scaled NumPy array
scipy.io.wavfile.write("techno.wav", rate=model.config.sampling_rate, data=output_np_scaled)

# scipy.io.wavfile.write("techno.wav", rate=model.config.sampling_rate, data=output)

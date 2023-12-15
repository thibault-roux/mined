import torch
from TTS.api import TTS

device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

# Init TTS with the target model name
tts = TTS(model_name="tts_models/multilingual/multi-dataset/xtts_v2").to(device)


multiples_texts = ["salut tu vas bien", "oui tu vas bien", "c'est super mon ami", "content de te voir", "également"]

i = 0
for text in multiples_texts:
    # Run TTS
    tts.tts_to_file(text, speaker_wav="files/bfm.wav", language="fr", file_path="files/bfm" + str(i) + ".wav")
    i += 1
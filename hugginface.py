# from transformers import pipeline

# classifier = pipeline("sentiment-analysis")

# result = classifier("Hugging Face makes NLP easy!")
# print(result)   


                        #TEXT TO SPEECH EXAMPLE USING HUGGING FACE TRANSFORMERS

# from transformers import pipeline
# import torch
# import soundfile as sf
# import numpy as np

# # Load TTS pipeline
# tts = pipeline(
#     "text-to-speech",
#     model="facebook/mms-tts-eng"
# )

# text = "Hello, this is a text to speech example using Hugging Face."

# # Generate speech
# output = tts(text)

# # Convert audio to numpy array
# audio = output["audio"]

# if isinstance(audio, torch.Tensor):
#     audio = audio.detach().cpu().numpy()

# # Ensure correct shape
# audio = np.squeeze(audio)

# # Write to WAV
# sf.write(
#     "output.wav",
#     audio,
#     output["sampling_rate"],
#     subtype="PCM_16"
# )

# print("✅ Audio saved as output.wav")




              #SENTIMENT ANALYSIS EXAMPLE USING HUGGING FACE TRANSFORMERS

from transformers import pipeline
from transformers import pipeline

pipe = pipeline("text-classification", model="distilbert/distilbert-base-uncased-finetuned-sst-2-english")
result=pipe("I love using Hugging Face transformers!")
print(result)

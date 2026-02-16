import torch
import soundfile as sf
from transformers import AutoTokenizer
from parler_tts import ParlerTTSForConditionalGeneration
from rubyinserter import add_ruby


class JapaneseParlerTTS:
    def __init__(self, device=None):
        self.device = device or ("cuda:0" if torch.cuda.is_available() else "cpu")

        print("Loading Parler-TTS model...")
        self.model = ParlerTTSForConditionalGeneration.from_pretrained(
            "2121-8/japanese-parler-tts-mini"
        ).to(self.device)

        self.prompt_tokenizer = AutoTokenizer.from_pretrained(
            "2121-8/japanese-parler-tts-mini",
            subfolder="prompt_tokenizer"
        )
        self.description_tokenizer = AutoTokenizer.from_pretrained(
            "2121-8/japanese-parler-tts-mini",
            subfolder="description_tokenizer"
        )

        self.description = (
            """A female speaker with a slightly high-pitched voice delivers her words at a moderate speed with a quite monotone tone in a confined environment, 
            resulting in a quite clear audio recording."""
        )

        # preload description tokens (quan trọng)
        self.description_ids = self.description_tokenizer(
            self.description, return_tensors="pt"
        ).input_ids.to(self.device)

        print("Parler-TTS ready.")

    @torch.inference_mode()
    def speak(self, text, output_wav="out.wav"):
        text = add_ruby(text)

        prompt_ids = self.prompt_tokenizer(
            text, return_tensors="pt"
        ).input_ids.to(self.device)

        generation = self.model.generate(
            input_ids=self.description_ids,
            prompt_input_ids=prompt_ids
        )

        audio = generation.cpu().numpy().squeeze()
        sf.write(output_wav, audio, self.model.config.sampling_rate)

        return output_wav

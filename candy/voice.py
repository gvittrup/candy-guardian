import hashlib
import logging
import subprocess
import wave
from pathlib import Path

from .config import resolve

log = logging.getLogger(__name__)


class VoiceBank:
    """Text -> spooky WAV, cached by text + backend + voice + effect."""

    def __init__(self, cfg):
        self.cfg = cfg
        self.backend = cfg.backend
        self.cache = resolve(cfg.cache_dir)
        self.cache.mkdir(parents=True, exist_ok=True)
        self.effect = [str(x) for x in cfg.effect]
        self._engine = None  # loaded on the first cache miss only

    def _voice_id(self):
        c = self.cfg
        if self.backend == "piper":
            return f"{Path(c.piper_model).name}:{c.length_scale}:{c.noise_scale}:{c.noise_w_scale}"
        if self.backend == "kokoro":
            return f"{c.kokoro_voice}:{c.kokoro_speed}"
        return f"{c.chatterbox_ref}:{c.chatterbox_exaggeration}:{c.chatterbox_cfg_weight}"

    def _key(self, text):
        raw = "|".join([text, self.backend, self._voice_id(), *self.effect])
        return hashlib.sha1(raw.encode()).hexdigest()[:16]

    def render(self, text):
        out = self.cache / f"{self._key(text)}.wav"
        if out.exists():
            return out
        dry = out.with_suffix(".dry.wav")
        getattr(self, f"_synth_{self.backend}")(text, dry)
        subprocess.run(["sox", str(dry), str(out), *self.effect], check=True)
        dry.unlink()
        return out

    def prerender(self, texts):
        for i, text in enumerate(texts, 1):
            self.render(text)
            if i % 10 == 0:
                log.info("rendered %d/%d lines", i, len(texts))

    def _synth_piper(self, text, path):
        from piper import PiperVoice, SynthesisConfig
        if self._engine is None:
            self._engine = PiperVoice.load(str(resolve(self.cfg.piper_model)))
        syn = SynthesisConfig(length_scale=self.cfg.length_scale,
                              noise_scale=self.cfg.noise_scale,
                              noise_w_scale=self.cfg.noise_w_scale)
        with wave.open(str(path), "wb") as wf:
            self._engine.synthesize_wav(text, wf, syn_config=syn)

    def _synth_kokoro(self, text, path):
        import numpy as np
        import soundfile as sf
        if self._engine is None:
            from kokoro import KPipeline
            # first letter of the voice name is the language: a = American, b = British
            self._engine = KPipeline(lang_code=self.cfg.kokoro_voice[0])
        chunks = [np.asarray(audio) for _, _, audio in
                  self._engine(text, voice=self.cfg.kokoro_voice, speed=self.cfg.kokoro_speed)]
        sf.write(str(path), np.concatenate(chunks), 24000)

    def _synth_chatterbox(self, text, path):
        import torchaudio
        if self._engine is None:
            from chatterbox.tts import ChatterboxTTS
            self._engine = ChatterboxTTS.from_pretrained(device="cpu")
        ref = str(resolve(self.cfg.chatterbox_ref)) if self.cfg.chatterbox_ref else None
        wav = self._engine.generate(text, audio_prompt_path=ref,
                                    exaggeration=self.cfg.chatterbox_exaggeration,
                                    cfg_weight=self.cfg.chatterbox_cfg_weight)
        torchaudio.save(str(path), wav, self._engine.sr)
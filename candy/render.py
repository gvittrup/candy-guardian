import logging

from . import config
from .lines import LineBook
from .voice import VoiceBank


def main():
    logging.basicConfig(level=logging.INFO)
    cfg = config.load()
    VoiceBank(cfg.voice).prerender(LineBook(cfg.costume.enabled).texts())


if __name__ == "__main__":
    main()
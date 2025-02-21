import os
import sys
import argparse
from pathlib import Path

from trainer import Trainer, TrainerArgs

FILE = Path(__file__).resolve()
ROOT = FILE.parents[3] #root directory
if str(ROOT) not in sys.path:
    sys.path.append(str(ROOT))  # add ROOT to PATH

from TTS.tts.configs.shared_configs import BaseDatasetConfig
from TTS.tts.configs.vits_config import VitsConfig
from TTS.tts.datasets import load_tts_samples
from TTS.tts.models.vits import CharactersConfig, Vits, VitsArgs, VitsAudioConfig
from TTS.tts.utils.languages import LanguageManager
from TTS.tts.utils.speakers import SpeakerManager
from TTS.tts.utils.text.tokenizer import TTSTokenizer
from TTS.utils.audio import AudioProcessor
from TTS.config import load_config

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description='Start training on Ruslan')    
    parser.add_argument('--continue_path', type=str, default="", help="Path to a training folder to continue training. Restore the model from the last checkpoint and continue training under the same folder.")    
    args = parser.parse_args()

    output_path = os.path.dirname(os.path.abspath(__file__))

    if args.continue_path:
        config = load_config(os.path.join(args.continue_path, "config.json"))
    else:

        ruslan_path = "RUSLAN"

        dataset_config = BaseDatasetConfig(formatter="ruslan", meta_file_train="metadata_RUSLAN_22200.csv", path=ruslan_path, language="ru-RU")

        audio_config = VitsAudioConfig(
            sample_rate=44100,
            win_length=1024,
            hop_length=256,
            num_mels=80,
            mel_fmin=0,
            mel_fmax=None,
        )

        config = VitsConfig(        
            audio=audio_config,
            run_name="vits_ruslan",        
            batch_size=16,
            eval_batch_size=8,
            batch_group_size=0,
            num_loader_workers=4,
            num_eval_loader_workers=4,
            run_eval=True,
            test_delay_epochs=-1,
            epochs=100,
            text_cleaner="multilingual_cleaners",
            use_phonemes=False,
            phoneme_language="ru-RU",
            phoneme_cache_path=os.path.join(output_path, "phoneme_cache"),
            compute_input_seq_cache=True,
            print_step=25,
            print_eval=False,
            mixed_precision=True,
            max_text_len=256,  # change this if you have a larger VRAM than 16GB
            output_path=output_path,
            datasets=[dataset_config],
            characters=CharactersConfig(
                characters_class="TTS.tts.models.vits.VitsCharacters",
                pad="<PAD>",
                eos="<EOS>",
                bos="<BOS>",
                blank="<BLNK>",
                characters="!'(),-–.:;?абвгдежзийклмнопрстуфхцчшщъыьэюяё ‘’‚“`”„",
                punctuations="!'(),–-.:;? ",
                phonemes=None,
            ),
            test_sentences=[
                ["Я думаю, что этот стартап действительно удивительный."],
            ],
        )

        # force the convertion of the custom characters to a config attribute
        config.from_dict(config.to_dict())

    # init audio processor
    ap = AudioProcessor(**config.audio.to_dict())

    # load training samples
    train_samples, eval_samples = load_tts_samples(
        config.datasets,
        eval_split=True,
        eval_split_max_size=config.eval_split_max_size,
        eval_split_size=config.eval_split_size,
    )

    # INITIALIZE THE TOKENIZER
    # Tokenizer is used to convert text to sequences of token IDs.
    # config is updated with the default characters if not defined in the config.
    tokenizer, config = TTSTokenizer.init_from_config(config)

    # init model
    model = Vits(config, ap, tokenizer, speaker_manager=None)

    # init the trainer and 🚀
    trainer = Trainer(
        TrainerArgs(args.continue_path), config, output_path, model=model, train_samples=train_samples, eval_samples=eval_samples
    )
    trainer.fit()

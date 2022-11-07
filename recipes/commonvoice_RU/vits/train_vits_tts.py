import os
import pandas as pd

from trainer import Trainer, TrainerArgs

from TTS.tts.configs.shared_configs import BaseDatasetConfig
from TTS.tts.configs.vits_config import VitsConfig
from TTS.tts.datasets import load_tts_samples
from TTS.tts.models.vits import CharactersConfig, Vits, VitsArgs, VitsAudioConfig
from TTS.tts.utils.languages import LanguageManager
from TTS.tts.utils.speakers import SpeakerManager
from TTS.tts.utils.text.tokenizer import TTSTokenizer
from TTS.utils.audio import AudioProcessor

output_path = os.path.dirname(os.path.abspath(__file__))

commonvoice_path = "cv-corpus-11.0-2022-09-21/ru"

def cv_formatter(root_path, manifest_file, **kwargs):    
    txt_file = os.path.join(root_path, manifest_file)
    df = pd.read_csv(txt_file,sep='\t')

    client_id2name = {c_id:f"cv_{i}" for i,c_id in enumerate(df["client_id"].unique())}
    df["client_id"] = df["client_id"].apply(lambda x: client_id2name[x])
    df["path"] = df["path"].apply(lambda x: os.path.join(root_path, "clips", x) )

    return df[["client_id","path","sentence"]].set_axis(["speaker_name","audio_file","text"],axis=1).to_dict(orient="records")


dataset_config = BaseDatasetConfig(formatter=cv_formatter, meta_file_train="validated.tsv", path=commonvoice_path, language="ru-RU")

audio_config = VitsAudioConfig(
    sample_rate=48000,
    win_length=1024,
    hop_length=256,
    num_mels=80,
    mel_fmin=0,
    mel_fmax=None,
)

vitsArgs = VitsArgs(
    use_speaker_embedding=True,
)

config = VitsConfig(
    model_args=vitsArgs,
    audio=audio_config,
    run_name="vits_cv_ru",
    use_speaker_embedding=True,
    batch_size=32,
    eval_batch_size=16,
    batch_group_size=0,
    num_loader_workers=4,
    num_eval_loader_workers=4,
    run_eval=True,
    test_delay_epochs=-1,
    epochs=1000,
    text_cleaner="russian_cleaners",
    use_phonemes=False,
    phoneme_language="ru-RU",
    phoneme_cache_path=os.path.join(output_path, "phoneme_cache"),
    compute_input_seq_cache=True,
    print_step=25,
    print_eval=False,
    mixed_precision=True,
    max_text_len=325,  # change this if you have a larger VRAM than 16GB
    output_path=output_path,
    datasets=dataset_config,
    characters=CharactersConfig(
        characters_class="TTS.tts.models.vits.VitsCharacters",
        pad="<PAD>",
        eos="<EOS>",
        bos="<BOS>",
        blank="<BLNK>",
        characters="!'(),-.:;?абвгдежзийклмнопрстуфхцчшщъыьэюяё ‘’‚“`”„",
        punctuations="!'(),-.:;? ",
        phonemes=None,
    ),
    test_sentences=[
        ["Я думаю, что этот стартап действительно удивительный.", "cv_206", None, "ru_RU"],
    ],
)

# force the convertion of the custom characters to a config attribute
config.from_dict(config.to_dict())

# init audio processor
ap = AudioProcessor(**config.audio.to_dict())

# load training samples
train_samples, eval_samples = load_tts_samples(
    dataset_config,
    eval_split=True,
    eval_split_max_size=config.eval_split_max_size,
    eval_split_size=config.eval_split_size,
)

# init speaker manager for multi-speaker training
# it maps speaker-id to speaker-name in the model and data-loader
speaker_manager = SpeakerManager()
speaker_manager.set_ids_from_data(train_samples + eval_samples, parse_key="speaker_name")
config.model_args.num_speakers = speaker_manager.num_speakers

# INITIALIZE THE TOKENIZER
# Tokenizer is used to convert text to sequences of token IDs.
# config is updated with the default characters if not defined in the config.
tokenizer, config = TTSTokenizer.init_from_config(config)

# init model
model = Vits(config, ap, tokenizer, speaker_manager)

# init the trainer and 🚀
trainer = Trainer(
    TrainerArgs(), config, output_path, model=model, train_samples=train_samples, eval_samples=eval_samples
)
trainer.fit()

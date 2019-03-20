import logging
import os
import shutil
import time
from argparse import ArgumentParser

from audio_reader import AudioReader
from constants import c
from utils import InputsGenerator


def arg_parse():
    #Read arguments
    arg_p = ArgumentParser()
    arg_p.add_argument('--audio_dir', required=True)
    arg_p.add_argument('--cache_output_dir', required=True)
    arg_p.add_argument('--regenerate_full_cache', action='store_true')
    arg_p.add_argument('--update_cache', action='store_true')
    arg_p.add_argument('--generate_training_inputs', action='store_true')
    arg_p.add_argument('--multi_threading', action='store_true')
    arg_p.add_argument('--unseen_speakers')  # p225,p226 example.
    arg_p.add_argument('--get_embeddings')  # p225 example.
    return arg_p


def regenerate_full_cache(audio_reader, args):
    cache_output_dir = os.path.expanduser(args.cache_output_dir)
    print('The directory containing the cache is {}.'.format(cache_output_dir))
    print('Going to wipe out and regenerate the cache in 5 seconds. Ctrl+C to kill this script.')
    time.sleep(5)
    try:
        shutil.rmtree(cache_output_dir)
    except:
        pass
    os.makedirs(cache_output_dir)
    audio_reader.build_cache()


def generate_cache_from_training_inputs(audio_reader, args):
    cache_dir = os.path.expanduser(args.cache_output_dir)
    inputs_generator = InputsGenerator(cache_dir=cache_dir,
                                       audio_reader=audio_reader,
                                       max_count_per_class=1000,
                                       speakers_sub_list=None,
                                       multi_threading=args.multi_threading)
    inputs_generator.start_generation()


def main():

    # Read arguments to use multi threading, define the output directory for the cache and audio directories
    args = arg_parse().parse_args()

    # Create an audio reader that returns a cache (dictionary) and metadata (dictionary)
    audio_reader = AudioReader(input_audio_dir=args.audio_dir,
                               output_cache_dir=args.cache_output_dir,
                               sample_rate=c.AUDIO.SAMPLE_RATE,
                               multi_threading=args.multi_threading)

    # Generate cache for the audio files. Caching usually involves sampling the WAV files at 8KHz and trimming the silences. 
    regenerate_full_cache(audio_reader, args)

    # Generate inputs used in the softmax training, MFCC windows randomly sampled from the audio cached files and put in a unified pickle file.
    generate_cache_from_training_inputs(audio_reader, args)

   


if __name__ == '__main__':
    logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')
    main()

import os
import json
root_path="../../Bench/data/levels/"
excluded_tasks=[]
model_name="X-Instruct-Blip-7B"

LLM_MODEL_PATH = "./vicuna"

import torch
import argparse
import numpy as np
from omegaconf import OmegaConf
from lavis.common.registry import registry
import random
from PIL import Image

def setup_seeds(seed=42):
    seed = seed

    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)

setup_seeds()
device = torch.device("cuda") if torch.cuda.is_available() else "cpu"

## Load Preprocessors
from lavis.processors.clip_processors import ClipImageEvalProcessor
from lavis.processors.audio_processors import BeatsAudioProcessor
from lavis.processors.alpro_processors import AlproVideoEvalProcessor

image_pocessor = ClipImageEvalProcessor()
audio_processor = BeatsAudioProcessor(model_name='iter3', sampling_rate=16000, n_frames=2, is_eval=False, frame_length=512)
video_processor = AlproVideoEvalProcessor(n_frms=4, image_size=224) # ! so less?

from lavis.models.blip2_models.blip2_vicuna_xinstruct import Blip2VicunaXInstruct
model = "vicuna7b_v2"
cfg_path  = {
        "vicuna13b": './configs/vicuna13b.yaml',
        "vicuna7b": './configs/vicuna7b.yaml',
        "no_init": './configs/vicuna7b_no_init.yaml',
        "projection": './configs/vicuna7b_projection.yaml',
        "vicuna7b_v2": './configs/vicuna7b_v2.yaml'
    }
    
config = OmegaConf.load(cfg_path[model].replace("./configs", "projects/xinstructblip/demo/configs"))
config.get("model", None).llm_model = LLM_MODEL_PATH
print('Loading model...')
model_cls = registry.get_model_class(config.get("model", None).arch)
model =  model_cls.from_config(config.get("model", None))
model.to(device)
print('Loading model done!')

def predict(
    prompt,
    image_list=None, 
    audio_list=None,
    video=None,
    qformer_prompt=None,
    min_len=1, 
    max_len=200, 
    beam_size=3, 
    len_penalty=-1., 
    repetition_penalty=1, 
    top_p=0.9, 
    decoding_method="Beam Search"
    ):
    if qformer_prompt == "" or qformer_prompt == None:
        qformer_prompt = prompt
    use_nucleus_sampling = decoding_method == "Nucleus sampling"
    
    image = None
    audio = None
    if image_list is not None:
        image = []
        for img in image_list:
            image.append(image_pocessor(Image.open(img)))
        image = torch.stack(image).unsqueeze(0).to(device) # shape (1, n_frames, 3, 224, 224)
    if audio_list is not None:
        audio = []
        for aud in audio_list:
            audio.append(audio_processor(aud))
        audio = torch.cat(audio, dim=0).unsqueeze(0).to(device)
    if video is not None:
        video = video_processor(video).unsqueeze(0).to(device)
    
    samples = {"prompt": prompt}
    if image is not None:
        samples["image"] = image
    if audio is not None:
        samples["audio"] = audio
    if video is not None:
        samples["video"] = video

    output = model.generate(
        samples,
        length_penalty=float(len_penalty),
        repetition_penalty=float(repetition_penalty),
        num_beams=beam_size,
        max_length=max_len,
        min_length=min_len,
        top_p=top_p,
        use_nucleus_sampling=use_nucleus_sampling,
        special_qformer_input_prompt=qformer_prompt
    )

    return output[0]


### dataset

# 本文件用于保存所有的定义
maic_cls_list = ['bus', 'hair-dryer', 'pipa', 'man', 'ambulance', 'razor', 'harp', 'tabla', 'bass', 'handpan', 
        'girl', 'sitar', 'car', 'lion', 'guitar', 'vacuum-cleaner', 'cat', 'mower', 'helicopter', 'boy', 'drum', 
        'keyboard', 'tuba', 'saw', 'flute', 'cello', 'woman', 'gun', 'accordion', 'violin', 'clarinet', 'erhu', 
        'saxophone', 'guzheng', 'dog', 'baby', 'horse', 'male', 'wolf', 'bird', 'ukulele', 'piano', 'female', 
        'marimba', 'not sure', 'no available option']

mvic_cls_list = ['sushi', 'banana', 'cake', 'butterfly', 'bird', 'microphone', 'hamburger', 'pineapple', 
        'man', 'book', 'sunglasses', 'goat', 'tie', 'cabinetry', 'motorcycle', 'drawer', 'strawberry', 
        'sheep', 'pasta', 'parrot', 'bull', 'table', 'penguin', 'watch', 'pillow', 'shellfish', 'kangaroo', 
        'flower', 'paddle', 'rocket', 'helicopter', 'bus', 'mushroom', 'bee', 'tree', 'boat', 'saxophone', 
        'football', 'lizard', 'violin', 'dog', 'cucumber', 'cello', 'airplane', 'horse', 'drum', 'box', 
        'rabbit', 'car', 'door', 'orange', 'shelf', 'camera', 'poster', 'lemon', 'cat', 'fish', 'bread', 
        'piano', 'apple', 'glasses', 'bicycle', 'truck', 'deer', 'woman', 'wheelchair', 'cheese', 'chair', 
        'plate', 'tomato', 'bed', 'starfish', 'balloon', 'bottle', 'crab', 'beer', 'frog', 'shrimp', 'tower', 
        'guitar', 'pig', 'peach', 'train', 'pumpkin', 'elephant', 'jellyfish', 'parachute', 'monkey', 'flag',
        'not sure', 'no available option']

prompt_avl = """
        In each video frame, there may be multiple categories of sound-emitting instances. Each category can have several instances. 
        You can choose instance categories from the given categories list.
        The naming format for instances is: category_id. Instance IDs start from 1, e.g., male_1, dog_2, dog_3, cat_4. 
        It is crucial that the instance names (i.e., category_id) remain consistent for the same instances across different frames.
        The bbox format is: [x, y, w, h], where x and y represent the coordinates of the top-left corner, and w and h are the width and height. 
        The final answer must strictly adhere to the following format: 
        answer={"frame_0": {"guzheng_1": "[269, 198, 83, 16]", "guzheng_2": "[147, 196, 75, 13]", "female_3": "[152, 108, 123, 36]"}, "frame_1": ..., "frame_n": ...}
    """

avl_cls_list = ['dog', 'clarinet', 'banjo', 'cat', 'guzheng', 'tree', 'lion', 'tuba', 
        'ukulele', 'flute', 'piano', 'person', 'violin', 'airplane', 'bass', 'pipa', 
        'trumpet', 'accordion', 'saxophone', 'car', 'lawn-mower', 'cello', 'bassoon', 
        'horse', 'guitar', 'erhu', 'not sure', 'no available option']

prompt_avlg = """
        Please output the answer in a format that exactly matches the following example:
        answer={'frame_0': [x0, y0, w0, h0], 'frame_1': None, ..., 'frame_9': [x9, y9, w9, h9]}.
        Note, for [x, y, w, h], where x and y represent the top-left corner of the bounding box, 
        and w and h represent the width and height of the bounding box.
    """

avqa_cls_list = ['ukulele', 'cello', 'clarinet', 'violin', 'bassoon', 'accordion', 'banjo', 'tuba', 'flute', 'electric_bass', 'bagpipe', 
        'drum', 'congas', 'suona', 'xylophone', 'saxophone', 'guzheng', 'trumpet', 'erhu', 'piano', 'acoustic_guitar', 'pipa', 'not sure', 'no available option']

havib_constants = {
    'L1_LAQA': {
        'options_sound_clarity': ['first', 'last', 'same', 'not sure'],
        'options_sound_order': ['sound', 'noise', 'not sure'],
        'options_sound_volume': ['first', 'last', 'same', 'not sure'],
        'options_yes_no': ['yes', 'no', 'not sure'],
    },

    'L1_LIQA': {
        'get_from_background_binary': ['yes', 'no', 'not sure'],
        'get_from_image_binary': ['yes', 'no', 'not sure'],
        'get_from_foreground_binary': ['yes', 'no', 'not sure'],
        'get_from_image_triple': ['blurred', 'normal', 'clear', 'not sure'],
        'get_from_3d-task1': ['center', 'left', 'right', 'not sure'],
        'get_from_3d-task2': ['cone', 'cube', 'cylinder', 'cuboid', 'no available option', 'not sure'],
        # 'get_from_3d-task3': [0, 1, 2, 3, 4, 5, 6],
        'get_from_space_hard': ['center', 'top left', 'top center', 'top right', 'bottom left', 'bottom center', 'bottom right', 'no available option', 'not sure'],
        'get_from_color': ['blue', 'green', 'red', 'puprle', 'yellow', 'no available option', 'not sure'],
        'get_yes_no': ['yes', 'no', 'not sure'],
        # 'get_lines_count': [0, 1, 2, 3, 4],
        'get_lines_direction': ['horizontal', 'vertical', 'inclined', 'not sure'],
        'get_from_space_easy_area': ['the right one', 'the left one', 'the middle one', 'the bottom one', 'the top one'],
        'get_from_space_easy_bbrightness': ['the right one', 'the left one', 'the middle one', 'the bottom one', 'the top one'],
        'options_yes_no': ['yes', 'no', 'not sure'],
    },

    'L1_LVQA': {
        'which_object': ['square', 'circle', 'triangle', 'not sure', 'no available option', 'not sure'],
        'what_shape': ['Triangular pyramid', 'Cone', 'Cube', 'Sphere', 'None', 'not sure'],
        # 'how_many': [0, 1, 2, 3, 4, 5, 6, 7, 8],
        'what_movement_2d': ['horizontal', 'inclined', 'vertical', 'no movenment', 'None', 'not sure'],
        'what_movement_3d': ['Rotation', 'Shrinking', 'Translation', 'Enlarging', 'None', 'not sure'],
        'what_surface': ['Rough', 'Moderate', 'Smooth', 'None', 'not sure'],
        'spacial_change': ['Bottom-left to top-right', 'Bottom-right to top-left', 'Top-left to bottom-right', 'Top-right to bottom-left', 'None', 'not sure', 'No movement',],
        'options_yes_no': ['yes', 'no', 'not sure'],
    },

    'L2_MAIC': {
        'maic_cls_list': maic_cls_list,
        'prompt_maic': "There may be one or more sound-emitting objects in the provided audio. \nPlease strictly output the answer in the format answer={'cls_1': count_1, 'cls_2': count_2}, \nfor example, answer={'dog': 2, 'cat': 3, 'male': 1}. \n"
    },

    'L2_MVIC': {
        'mvic_cls_list': mvic_cls_list,
        'prompt_mvic': "There may be one or more visible objects in the provided image. \nPlease strictly output the answer in the format answer={'cls_1': count_1, 'cls_2': count_2}, \nfor example, answer={'dog': 2, 'cat': 3, 'male': 1}. \n Possible categoris are in the list: mvic_cls_list"
    },

    'L3_AVH': {
        'prompt_avh': "Please answer the question based on the given audio and video.",
        'avh_options_list': ['yes', 'no', 'not sure'],
    },

    'L3_VAH': {
        'prompt_vah': "Please answer the question based on the given audio and video.",
        'vah_options_list': ['yes', 'no', 'not sure'],
    },

    'L3_AVL': {
        'prompt_avl': prompt_avl,
        'avl_cls_list': avl_cls_list,
    },

    'L3_AVM': {
        'prompt_avm': 'Please answer the question based on the given audio and video.',
        'avm_options_list': ['yes', 'no', 'not sure'],
    },

    'L3_AVR': {
        'prompt_avr': "Please output the indices of the images list, starting from 0. For example: [], or [0, 3], or [1, 4, 9]."
    },

    'L3_VAR': {
        'prompt_var': "Please output the indices of the wavs list, starting from 0. For example: [], or [0, 3], or [1, 4, 9]."
    },

    'L4_AVC': {

    },

    'L4_AVLG': {
        'prompt_avlg': prompt_avlg,
    },

    'L4_AVQA': {
        'avqa_options_list_is': ['yes', 'no', 'not sure'],
    },

    'L5_AVLG': {
        'prompt_avlg': prompt_avlg,
    },

    'L5_AVQA': {
        'avqa_cls_list': avqa_cls_list,
        'avqa_options_list_is': ['yes', 'no', 'not sure'],
    },
}

def get_real_options_or_classes(d: dict):
    """ replace the pseudo-options with real options. """

    if 'options' in d['input']['question'].keys():
        options = d['input']['question']['options']

        if options in havib_constants[d['task']].keys():  
            options = havib_constants[d['task']][options]

        if options is not None:
            if 'cls' in options:
                opt_or_cls = 'semantic categories'
            else:
                opt_or_cls = 'options'

            options = f'Available {opt_or_cls} are: {options}'
        else: 
            options = ''
    else:
        options = ''
    
    return options

def get_real_prompt(d: dict):
    """ replace the pseudo-prompt with real prompt. """

    prompt = ''
    if 'prompt' in d['input']['question'].keys():
        prompt = d['input']['question']['prompt']
        
        if prompt in havib_constants[d['task']].keys():  # replace the pseudo-options with real options.
            prompt = havib_constants[d['task']][prompt]

        if prompt is None:
            prompt = ''
    else:
        prompt = ''
    
    return prompt

def get_real_input(d: dict):
    """ concat input info: text_input = prompt + options + question. """
    prompt = get_real_prompt(d)  # replace the pseudo-prompt with real prompt.
    options = get_real_options_or_classes(d)  # replace the pseudo-options with real options.
    question = d['input']['question']['text']
    text_input = f'{prompt}. {options}. {question}'
    
    return text_input

def run_task(task_path, task_name):
    save_prediction_json = f'./results/{model_name}/tasks/{task_name}.json'
    if os.path.exists(save_prediction_json):
        print(f'>>> {save_prediction_json} already exists, skip.')
        return
    else:
        print(">>> " + task_name + " ...")
    os.makedirs(os.path.dirname(save_prediction_json), exist_ok=True)

    dataset = json.load(open(os.path.join(task_path, "data.json"), "r"))
    predictions = []
    for idx in range(len(dataset)):
        print(f'>>> idx: {idx} | #tot={len(dataset)}', end='\r')
        data = dataset[idx]
        # print('>>>', data)

        _id = data['id']        # ! do not change
        _task = data['task']    # ! do not change
        assert _task == task_name, f'Evaluating {task_name} but found: {_task}'

        text_input = get_real_input(data)  # real_input = prompt + options + question
        audio_list = data['input'].get('audio_list', None)
        image_list = data['input'].get('image_list', None)
        video = data['input'].get('video', None)

        # replace data src prefix.
        if audio_list is not None:
            for idx, aud in enumerate(audio_list):
                audio_list[idx] = aud.replace('./input', f'{task_path}/input')
                assert os.path.exists(audio_list[idx]), f'Not found - audio: {audio_list[idx]}'

        if image_list is not None:
            for idx, img in enumerate(image_list):
                image_list[idx] = img.replace('./input', f'{task_path}/input')
                assert os.path.exists(image_list[idx]), f'Not found - image: {image_list[idx]}'

        if video is not None:
            video = video.replace('./input', f'{task_path}/input')
            assert os.path.exists(video), f'Not found - video: {image_list[idx]}'
        
        # Early attempts at better debugging.
        pred_ans = predict(text_input, image_list, audio_list, video)
        
        pred_record = {
            "task": _task,
            "id": _id,
            "predict": pred_ans,
        }
        predictions.append(pred_record)

        # print('>>> ans=:', pred_record)

    # save the predictions
    with open(save_prediction_json, 'w', encoding='utf-8') as json_file:
        json.dump(predictions, json_file, ensure_ascii=False, indent=4)

if __name__ == '__main__':

    levels = [d for d in os.listdir(root_path) if os.path.isdir(os.path.join(root_path, d))]
    for level in levels:
        tasks = os.listdir(os.path.join(root_path, level))
        for task in tasks:
            task_path = os.path.join(root_path, level, task)
            task_name = f"L{task_path.rsplit('/', 1)[0][-1]}_{task_path.rsplit('/', 1)[-1]}"  # L1_LAQA
            if task_name in excluded_tasks:
                print(f'>>> {task_name} is excluded.')
                continue
            run_task(task_path, task_name)
            print(f'>>> {task_name} done.')
    
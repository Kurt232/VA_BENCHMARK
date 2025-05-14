import os
import json
from definitions import *
root_path="/home/jiangli/wjdu/data3/data"
excluded_tasks=[]
save_path="../wjdu0512/"
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
video_processor = AlproVideoEvalProcessor(n_frms=10, image_size=224) # ! so less?

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
    max_len=512, 
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
    save_prediction_json = f'{save_path}/{model_name}/tasks/{task_name}.json'
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
        if video is not None:
            video = None if "NO_USE" in video else video
        # replace data src prefix.
        if audio_list is not None:
            for idx, aud in enumerate(audio_list):
                if "NO_USE" in aud:
                    audio_list = None
                    break
                audio_list[idx] = aud.replace('./input', f'{task_path}/input')
                assert os.path.exists(audio_list[idx]), f'Not found - audio: {audio_list[idx]}'

        if image_list is not None:
            for idx, img in enumerate(image_list):
                if "NO_USE" in img:
                    image_list = None
                    break
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
    
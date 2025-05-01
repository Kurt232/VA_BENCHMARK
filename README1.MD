## Installation

### LAVIS Repository
```
# python 3.8
pip install -e .
pip install -r requirements.txt
```

## Pretrained Models
### Language Model Weights
```
huggingface-cli download lmsys/vicuna-7b-v1.1

ln -s $MODEL_PATH ./vicuna
```

### X-InstructBLIP Weights
```
python bench.py # auto loading
```
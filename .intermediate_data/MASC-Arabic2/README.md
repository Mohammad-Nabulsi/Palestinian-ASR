---
dataset_info:
  features:
  - name: video_id
    dtype: string
  - name: duration
    dtype: float64
  - name: text
    dtype: string
  - name: type
    dtype: string
  - name: audio
    dtype:
      audio:
        sampling_rate: 16000
  splits:
  - name: train
    num_bytes: 199910559219
    num_examples: 875873
  - name: validation
    num_bytes: 4798320986
    num_examples: 19521
  - name: test
    num_bytes: 4477186626
    num_examples: 18006
  download_size: 184871731199
  dataset_size: 209186066831
configs:
- config_name: default
  data_files:
  - split: train
    path: data/train-*
  - split: validation
    path: data/validation-*
  - split: test
    path: data/test-*
task_categories:
- automatic-speech-recognition
- text-to-speech
language:
- ar
pretty_name: MASC Arabic
license: cc-by-4.0
---


# MASC Arabic Dataset Card

## Dataset Description

- **Homepage:** https://ieee-dataport.org/open-access/masc-massive-arabic-speech-corpus
- **Paper:** https://ieeexplore.ieee.org/document/10022652
- **Original Dataset Repo:** https://huggingface.co/datasets/pain/MASC

### Dataset Summary

MASC is a dataset that contains 1,000 hours of speech sampled at 16 kHz and crawled from over 700 YouTube channels.
The dataset is multi-regional, multi-genre, and multi-dialect intended to advance the research and development of Arabic speech technology with a special emphasis on Arabic speech recognition.

## How to use

The `datasets` library allows you to load and pre-process your dataset in pure Python, at scale. The dataset can be downloaded and prepared in one call to your local drive by using the `load_dataset` function. 

```python
from datasets import load_dataset

masc = load_dataset("MohamedRashad/MASC-Arabic", split="train")
```

Using the datasets library, you can also stream the dataset on-the-fly by adding a `streaming=True` argument to the `load_dataset` function call. Loading a dataset in streaming mode loads individual samples of the dataset at a time, rather than downloading the entire dataset to disk.
```python
from datasets import load_dataset

masc = load_dataset("MohamedRashad/MASC-Arabic", split="train", streaming=True)

print(next(iter(masc)))
```

*Bonus*: create a [PyTorch dataloader](https://huggingface.co/docs/datasets/use_with_pytorch) directly with your own datasets (local/streamed).

### Local

```python
from datasets import load_dataset
from torch.utils.data.sampler import BatchSampler, RandomSampler

masc = load_dataset("MohamedRashad/MASC-Arabic", split="train")
batch_sampler = BatchSampler(RandomSampler(masc), batch_size=32, drop_last=False)
dataloader = DataLoader(masc, batch_sampler=batch_sampler)
```

### Streaming

```python
from datasets import load_dataset
from torch.utils.data import DataLoader

masc = load_dataset("MohamedRashad/MASC-Arabic", split="train")
dataloader = DataLoader(masc, batch_size=32)
```

To find out more about loading and preparing audio datasets, head over to [hf.co/blog/audio-datasets](https://huggingface.co/blog/audio-datasets).

## Dataset Structure

### Data Instances

A typical data point comprises the `path` to the audio file and its `sentence`. 

```python
{'video_id': 'OGqz9G-JO0E', 'duration': 11.24,
'text': 'اللهم من ارادنا وبلادنا وبلاد المسلمين بسوء اللهم فاشغله في نفسه ورد كيده في نحره واجعل تدبيره تدميره يا رب العالمين',
'type': 'c',
'audio': {'path': None,
    'array': array([
                0.05938721,
                0.0539856,
                0.03460693, ...,
                0.00393677,
                0.01745605,
                0.03045654
            ]),
    'sampling_rate': 16000
    }
}
```

### Data Fields

- **video_id**: An id for the video that the voice has been created from
- **duration**: The duration of the chunk
- **text**: The text of the chunk
- **type**: It refers to the data set type, either clean or noisy where "c: clean and n: noisy"
- **audio**: Audio for the chunk

### Data Splits

The speech material has been subdivided into portions for train, dev, test.

The dataset splits has clean and noisy data that can be determined by type field.


### Citation

```
@INPROCEEDINGS{10022652,
  author={Al-Fetyani, Mohammad and Al-Barham, Muhammad and Abandah, Gheith and Alsharkawi, Adham and Dawas, Maha},
  booktitle={2022 IEEE Spoken Language Technology Workshop (SLT)}, 
  title={MASC: Massive Arabic Speech Corpus}, 
  year={2023},
  volume={},
  number={},
  pages={1006-1013},
  doi={10.1109/SLT54892.2023.10022652}}
}
```
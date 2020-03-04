# Model for Banana Project

The model will classify if a banana is good or flecked.

## Install

```bash
pip3 install -r requirements.txt
```

## Rebuild model

The dataset used is not uploaded.

The model is trained with 4848 pictures in the training set and 1488 pictures in the validation set.

The size of an input picture is 150x150 pixels (the model reshape the pictures).

```bash
cd Train
python3 train.py
```

## Test

```bash
python3 t_main.py
```
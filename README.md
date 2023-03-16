

# Bart-coreml

This is an experimental repo to convert [Bart](https://github.com/NVIDIA/DeepLearningExamples/tree/master/PyTorch/LanguageModeling/BART) model to CoreML.

* We have to convert two decoder model, one for first iteration and one for the rest of iterations.
* We can't use fixed shape for inputs 
* This repository is not clean due to lots of trial and error 😆

see also:

Run encoder on Apple Neural Engine #548  
https://github.com/ggerganov/whisper.cpp/discussions/548


# Env

## python 

[Python 3.10.9](https://www.python.org/downloads/macos/)


### coremltools

verify uninstallation of existing ANE-related modules that may conflict with coremltools.

```
pip3 uninstall tensorboard
pip3 uninstall tensorflow-metal
pip3 uninstall tensorflow-macos
pip3 uninstall ane-transformers
```

coremltools 6.2 supports pytorch 1.13.1 with numpy 1.24.0

```
pip3 uninstall numpy
pip3 install numpy==1.24.0

pip3 uninstall torch torchvision
pip3 install torch==1.13.1 torchvision==0.14.1
```

install dependencies

```
pip3 install packaging
pip3 install protobuf
pip3 install google
pip3 install google-api-python-client
pip3 install shutils
```

build coremltools

```
cd coremltools
zsh -i scripts/build.sh --python=3.10.9 --dist

cd build/dist
rm coremltools-6.1-cp310-none-macosx_10_15_x86_64.whl
pip3 uninstall coremltools
pip3 install coremltools-6.2-cp310-none-macosx_10_15_x86_64.whl
```

## bart dependencies

for nvidia bart

```
pip3 install tokenizers
pip3 install filelock
pip3 install nvidia-pyindex
pip3 install nvidia-dllogger
pip3 install nvidia-ml-py
pip3 install psutil
pip3 install nlp
pip3 install py3nvml
pip3 install apex
pip3 install GitPython
pip3 install fairseq
```

# source model

put the target model on the same level of this repo

```
cd ..
cd HuggingFaceModels
git clone https://huggingface.co/sshleifer/distilbart-xsum-12-3
```

# convert model

```
cd Bart-coreml
python convert.py ../HuggingFaceModels/distilbart-xsum-12-3
```

# evaluate model

```
python cml_beam_search.py <path>/<to>/<summarize target>.txt
```

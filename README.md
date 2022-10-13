# Microstructure Inpainter

<p align="center">
<img src="assets/movie.gif" alt="inpainting-movie" width="400px"/>
</p>

Microstructure inpainter is a python app for inpainting material science microstructural images using GANs. The app can be run via a command line interface or a graphical interface. Although the GANs can be trained on CPU, GPU training is advised.

## Quickstart

To run from executable file. Download the file specific for your operating system.

MacOS users - unzip, and then double click the icon and allow access to folders.

Linux users - unzip and then run from command line

```
./path/to/inpainter
```

To run locally.

Prerequisites:

- conda
- python3

Create a new conda environment, activate and install pytorch

_Note: cudatoolkit version and pytorch install depends on system, see [PyTorch install](https://pytorch.org/get-started/locally/) for more info._

```
conda create --name inpainter
conda activate inpainter
conda install pytorch torchvision -c pytorch
pip install -r requirements.txt
```

If you are planning to use Weights and Biases to track training runs create a .env file to hold secrets, the .env file must include

```
WANDB_API_KEY=
WANDB_ENTITY=
WANDB_PROJECT=
```

You are now ready to run the repo. To run the GUI

```
python run_gui.py
```

Alternatively, to start training a model using the CLI, use the following command

```
python run_cli.py train -t <name> -c <x1 x2 y1 y2> -p <path/to/data> -i <image_type> -s <method> -w <WANDB>
```

`-t` takes the name of the training run, `-c` takes the coordinate values as a vector of 4 numbers representing the corners of a rectangle, `-p` takes the path to the data, `-i` takes the image type `(n-phase, grayscale, colour)`, `-s` takes the method name `(rect, poly)`, `-w` takes a boolean whether to track run with Weights and Biases. For example,

```
python run_cli.py train -t test -c <360 440 360 440> -p <data/example_inpainting.png> -i n-phase -s rect -w False
```

To generate samples from a trained generator

```
python run_cli.py generate -t test -c <360 440 360 440> -p <data/example_inpainting.png> -i n-phase -s rect -w False
```

## Saving, loading and overwriting models

Models are saved to runs folder which is generated when training initiates. Inside runs, a new folder with the name of your training run tag will be generated, inside this the model params and training outputs are saved. This includes:

- **config.json** - this json holds the config paramaters of your training run, see config.py for more info
- **Gen.pt** - this holds the generator training parameters
- **Disc.pt** - this holds the discriminator parameters

### Training

If training for the first time, these files are created and updated during training.

If you initiate a training run with a tag of a run that already exists you will see the prompt

```
To overwrite existing model enter 'o', to load existing model enter 'l' or to cancel enter 'c'.
```

By entering `'o'` you will overwrite the existing models, deleting their saved parameters and config. `l` will load the existing model params and config, and continue training this model. `c` will abort the training run.

### Evaluation

When evaluating a trained model, the params and model config are loaded from files. Models are saved with their training tag, use this tag to evaluate specific models.

## Roadmap

[ ] Windows single exe
[ ] Save and load trained models
[ ] Implement both methods for both shapes

# Galaxy Zoo generation using DDPM

## Installation
```bash
conda env create -f environment.yml
```

## Dataset
[Galaxy Zoo](https://www.kaggle.com/c/galaxy-zoo-the-galaxy-challenge) dataset is used

## Training
### Training of the classifier (used for guidance)
To run the training of the classifier, first fill the **config file**. Example of the detailed config is available `configs/classifier.yaml`

Then run:
```bash
python train_classifier.py --config=<path to config>
```

If you want to automatically select the batch size, add `--auto_bs` flag. If you want to automatically select learning rate, add `--auto_lr` flag.

### Training of the conditional DDPM generator
To run the training of the classifier, first fill the **config file**. Example of the detailed config is available `configs/generator.yaml`

Then run
```bash
python train_generator.py --config=<path to config>
```

If you want to automatically select the batch size, add `--auto_bs` flag. If you want to automatically select learning rate, add `--auto_lr` flag.

## Image generation
### Generate images using classifier guidance

Run
```bash
python classifier_sample.py --config_gen=<generator config> \
                            --config_clas=<classifier config> \
                            --ckpt_gen=<generator ckpt> \
                            --ckpt_clas=<classifier ckpt> \
                            --classifier_scale=3 \
                            --batch_size=16 \
                            --output=<output path> \
                            --timestep_respacing=250 
```

### Generate images using classifier-free guidance

Run
```bash
python classifier_free_samlpe.py --config=<generator config> \
                                 --ckpt=<generator ckpt> \
                                 --output=<path where to save generated images> \
                                 --batch_size=16 \
                                 --guidance_scale=3 \
                                 --timestep_respacing=250
                                 
```

## Evaluation

To run the generated images evaluation
```bash
python evaluate.py --path_data=<path to real images directory> \
                   --path_labels=<path to csv with labels> \
                    --path_gen_images=<path to .npy file with generated images> \ 
                    --path_gen_labels=<path to .npy file with labels used to generate images>
```
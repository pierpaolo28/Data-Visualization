# Weights & Biases Tutorial
Get started tracking your deep learning training with this handy sample code. This simple training script uses a convolutional neural network to classify clothing items from photos. If you'd like to learn more about our company, check our our [blog](wandb.com/blog), and we're always available to provide [live demos](wandb.com/contact).

1. Make sure you have W&B installed.
```
pip install wandb
wandb signup
```

2. Initialize our sample script.
```
git clone http://github.com/cvphelps/tutorial
cd tutorial
pip install -r requirements.txt
wandb init
```

3. Run the CNN script from the command line.
```
python tutorial.py
```

Awesome! Now head over to [Weights & Biases](https://app.wandb.ai) to visualize your model training and create a new project of your own.

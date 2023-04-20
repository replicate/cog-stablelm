# Cog StableLM

[![Replicate](https://replicate.com/replicate/stablelm/badge)](https://replicate.com/replicate/stablelm)

This repository is an implementation of [StableLM](https://github.com/Stability-AI/StableLM) as a Cog model. [Cog packages machine learning models as standard containers.](https://github.com/replicate/cog).

## Prerequisites

* Model weights.

* GPU machine. You'll need a Linux machine with an NVIDIA GPU attached and the NVIDIA Container Toolkit installed. If you don't already have access to a machine with a GPU, check out our guide to getting a GPU machine. This codebase currently assumes a single device with sufficient VRAM (>24GB) is available. If, instead, you have access to a multi-device environment, you can modify the code to distribute your model across devices.

* Docker. You'll be using the Cog command-line tool to build and push a model. Cog uses Docker to create containers for models.

## Step 0: Install Cog

First, [install Cog](https://github.com/replicate/cog#install):

```
curl -o /usr/local/bin/cog -L "https://github.com/replicate/cog/releases/download/v0.7.0-beta17/cog_Linux_x86_64"
sudo chmod +x /usr/local/bin/cog
```

## Step 1: Run the model


You can run the model locally to test it:

```
cog predict -i prompt="What is a meme, and what's the history behind this word?"
```

## Step 2: Create a model on Replicate

Go to [replicate.com/create](https://replicate.com/create) to create a Replicate model.

Make sure to specify "private" to keep the model private.

## Step 3: Configure the model to run on A100 GPUs

Replicate supports running models on a variety of GPUs. The default GPU type is a T4, but for best performance you'll want to configure your model to run on an A100.

Click on the "Settings" tab on your model page, scroll down to "GPU hardware", and select "A100". Then click "Save".

## Step 4: Push the model to Replicate

Log in to Replicate:

```
cog login
```

Push the contents of your current directory to Replicate, using the model name you specified in step 3:

```
cog push r8.im/username/modelname
```

[Learn more about pushing models to Replicate.](https://replicate.com/docs/guides/push-a-model)


## Step 5: Run the model on Replicate

Now that you've pushed the model to Replicate, you can run it from the website or with an API.

To use your model in the browser, go to your model page.

To use your model with an API, click on the "API" tab on your model page. You'll see commands to run the model with cURL, Python, etc.

To learn more about how to use Replicate, [check out our documentation](https://replicate.com/docs).

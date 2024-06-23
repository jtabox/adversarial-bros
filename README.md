### 👨🏻👨🏼👨🏽👨🏾👨🏿 Adversarial Bros 👨🏿👨🏾👨🏽👨🏼👨🏻

**A simple (my ass) Gradio app that generates face photos of delicious imaginary bros using BroGAN, a StyleGAN3 model.**

It can generate and save single or bulk generate multiple images with different seeds and other settings.

I'm also planning some extra functionality for a later point in the future (if I don't get bored and abandon this), see ToDos below.

###### Info:

* I wrote this because I wanted to be able to use BroGAN without firing up A1111 and use [the relevant extension for it](https://github.com/zerxiesderxies/sd-webui-gan-generator/), and without having to use the terminal and cli. Then I got interested in getting some more experience in Gradio and Python programming in general, so I kinda spent more time with it.
* It's still an exceptionally messy, probably buggy patchwork of mostly other people's code and a bit of my own (= I asked various LLMs to do stuff). But it somehow works?

**Caution: It executes pickle code because it has to, and generally I give literally zero guarantees about the whole thing's safety and proper behaviour. Likewise I accept zero responsibility. So please read the code and use at your own risk, or don't.**

###### Requirements:

* An NVIDIA card with CUDA support (can work with CPU if you change stuff in the code).
* A Python environment (I used 3.10 but others may work I guess) with uuuuh... `gradio`, `torch`, `pickle` and `pillow` and... `diffusers`? I might've completely failed to take a note of the packages I installed, so I don't remember all of them. Install them if it complains.
* It'll need to compile some CUDA binaries at some point, so you need `nvcc` and `Visual Studio Build Tools`.
* The whole thing was developed and run on Windows 11 but it should be relatively easy to run in Linux with minimal adaptations. I think.

###### Usage:

* Download BroGAN from [https://huggingface.co/quartzermz/BroGANv1.0.0](https://huggingface.co/quartzermz/BroGANv1.0.0) and put it in the `model` folder.
* Run `python bros.py` or `gradio bros.py`.
* Use it. There's instructions inside for the various functions and settings.

###### ToDos:

* Implement CodeFormer integration so the images can be upscaled from the default 256x256 px.

###### Credits:

- Huggingface user [**quartzermz**](https://huggingface.co/quartzermz), who did God's work and created [**BroGAN**](https://huggingface.co/quartzermz/BroGANv1.0.0), a StyleGAN3 model for PyTorch.
- [**NVlabs**](https://github.com/NVlabs/stylegan3) at NVIDIA, creators of the StyleGAN3 model that BroGAN is based on, and GitHub user [**PDillis**](https://github.com/PDillis) and their stylegan3 fork, [**stylegan3-fun**](https://github.com/PDillis/stylegan3-fun) whose code I've used as a backend here.
- GitHub user [**ZerxiesDerxies**](https://github.com/zerxiesderxies), author of the very helpful rentry.co guide [**'A men-only SD Webui Guide to running BroGAN'**](https://rentry.co/uoza8fqp), with a lot of info about BroGAN. And also author of the [**relevant A1111 extension**](https://github.com/zerxiesderxies/sd-webui-gan-generator/), which was the initial inspiration for this project.
- The devs of [**CodeFormer**](https://github.com/sczhou/CodeFormer), which is used to upscale the photos in this project.
- Users at the **#men-only channel** on the [**Unstable Diffusion Discord server**](https://discord.gg/unstablediffusion), by who I was made aware of BroGAN.

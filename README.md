### Adversarial Bros

**A simple (my ass) Gradio app that generates face photos of delicious (albeit imaginary) bros using BroGAN, a StyleGAN3 model.**

I wrote this because I wanted to be able to use BroGAN without firing up A111 (since there's an extension for it) and without having to use the CLI. It's messy, buggy and a patchwork of other people's code, but it somehow works?

**It executes pickle code and generally I give literally zero guarantees about the whole thing's safety and behaviour, so please read the code and use at your own risk or don't.**


###### Requirements:

* An NVIDIA card with CUDA support (can work with CPU if you change stuff in the code).
* A Python environment (I used 3.10 but others may work I guess) with `gradio`, `torch`, `pickle` and `pillow` (might've installed a couple more that I don't remember right now, install them if it complains).
* It'll need to compile some CUDA binaries at some point, so you need `nvcc` and `Visual Studio Build Tools`.

###### Usage:

* Download BroGAN from [https://huggingface.co/quartzermz/BroGANv1.0.0](https://huggingface.co/quartzermz/BroGANv1.0.0) and put it in the `model` folder.
* Run `bros.py` (with either `python` or `gradio` command).
* Use it.

###### Credits:

- Huggingface user [**quartzermz**](https://huggingface.co/quartzermz), who did God's work and created [**BroGAN**](https://huggingface.co/quartzermz/BroGANv1.0.0), a StyleGAN3 model for PyTorch.
- [**NVlabs**](https://github.com/NVlabs/stylegan3) at NVIDIA, who created the StyleGAN3 model that BroGAN is based on, and whose code is used in this app as a backend.
- GitHub user [**ZerxiesDerxies**](https://github.com/zerxiesderxies), author of the very helpful rentry.co guide [**'A men-only SD Webui Guide to running BroGAN'**](https://rentry.co/uoza8fqp), with a lot of info about BroGAN.
- Users at the **#men-only channel** on the [**Unstable Diffusion Discord server**](https://discord.gg/unstablediffusion), by who I was made aware of BroGAN.

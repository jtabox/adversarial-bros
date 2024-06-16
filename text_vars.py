# Some text variables for the main script

title = "# 👨🏻👨🏼👨🏽👨🏾👨🏿 Adversarial Bros 👨🏿👨🏾👨🏽👨🏼👨🏻"

description = "A simple (yeah right) Gradio app that generates face photos of delicious (albeit imaginary) bros using BroGAN, a StyleGAN3 model."

info_text = """
<p>🔵 Start by choosing the image generation mode by selecting the appropriate tab. At the moment, there's two modes:</p>
&emsp;&emsp;<b><u>Simple Image Generator:</u></b> Generates and displays a single image with a seed and truncation psi value of your choice. Used mainly for experimenting with the model's different parameters, figure out how they affect the result, etc.<br>
&emsp;&emsp;<b><u>Bulk Generator:</u></b> Not yet implemented, but will allow generating and saving multiple images from various seeds in bulk.<br>
<br>
<p>🔵 The generated images are in PNG format and have a size of 256x256 pixels.</p><br>
<p>🔵 The available parameters are (in ignorant layman's terms):</p>
&emsp;&emsp;<b><u>Seed:</u></b> Each seed is an integer and creates a unique face. There's almost <i>4.3 billion</i> seeds available for your visual pleasure.<br>
&emsp;&emsp;<b><u>Truncation psi:</u></b> A decimal number ranging from -1 to 1. Values further away from 0 create more diverse faces but with worse quality. The recommended value is 0.7, but feel free to experiment.<br>
<br>
<p>🔵 For more information about the model and other things, check the Credits section at the bottom of the page.</p>
"""

credits_text = """
- Huggingface user [**quartzermz**](https://huggingface.co/quartzermz), who did God's work and created [**BroGAN**](https://huggingface.co/quartzermz/BroGANv1.0.0), a StyleGAN3 model for PyTorch.
- [**NVlabs**](https://github.com/NVlabs/stylegan3) at NVIDIA, who created the StyleGAN3 model that BroGAN is based on, and whose code is used in this app as a backend.
- GitHub user [**ZerxiesDerxies**](https://github.com/zerxiesderxies), author of the very helpful rentry.co guide [**'A men-only SD Webui Guide to running BroGAN'**](https://rentry.co/uoza8fqp), with a lot of info about BroGAN.
- Users at the **#men-only channel** on the [**Unstable Diffusion Discord server**](https://discord.gg/unstablediffusion), by who I was made aware of BroGAN (and other  *t a s t y*  #men-only generative-AI stuff 😏).
"""
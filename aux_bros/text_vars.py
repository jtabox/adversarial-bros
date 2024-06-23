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
&emsp;&emsp;<b><u>Truncation psi:</u></b> A decimal number ranging from 0 to 1. Values further away from 0 create more diverse faces but with worse quality. The recommended value is 0.7, but feel free to experiment.<br>
&emsp;&emsp;<b><u>Negative psi:</u></b> The psi value above, but negative. Apparently the same value gives 2 different images when positive and negative. So this gives 2 options per psi value.<br>
<br>
<p>🔵 For more information about the model and other things, check the Credits section at the bottom of the page.</p>
"""

bulk_info_text = """
<p>The following 2 alternatives are available for generating multiple images:</p><br>
<p>🔵 <u>Generate a specific amount of random images (default, 10 images):</u><br>
<ul>
<li>Set #1 to the desired amount. Must be a positive integer.</b></li>
<li>Any potential settings in #2 and #3 will be ignored.</li>
<li>A random seed from the whole available range (0-4294967295) and a psi value of 0.65 will be used for each image.</li>
<li>Example: setting #1 to '10' will use 10 random seeds from 0 to 4294967295 and will create 1 image for each seed, resulting in 10 images total.</li>
</ul></p>
<br>
<p>🔵 <u>Use specific seeds, psi values or a combination of both:</u><br>
<ul>
<li>Set #1 to 0 or blank.</li>
<li>Set the desired seeds in #2. Must be integers ranging from 0 to 4294967295 and in ascending order.</li>
<li>Set #3 to the desired psi value(s). Must be decimals ranging from -1.00 to 1.00, more decimal precision points will be rounded to 2 points.</li>
<li>If #3 is left blank or doesn't contain any valid decimal in range, a psi value of 0.65 will be used for all the images.</li>
<li>Example: setting #2 to '0-2, 20, 35-37' and #3 to '-0.65, 0.7' will use seeds 0, 1, 2, 20, 35, 36, 37 and will create 2 images for each seed, one with psi -0.65 and one with 0.7, resulting in 14 images total.</li>
</ul></p>
"""

credits_text = """
- Huggingface user [**quartzermz**](https://huggingface.co/quartzermz), who did God's work and created [**BroGAN**](https://huggingface.co/quartzermz/BroGANv1.0.0), a StyleGAN3 model for PyTorch.
- [**NVlabs**](https://github.com/NVlabs/stylegan3) at NVIDIA, who created the StyleGAN3 model that BroGAN is based on, and whose code is used in this app as a backend.
- GitHub user [**ZerxiesDerxies**](https://github.com/zerxiesderxies), author of the very helpful rentry.co guide [**'A men-only SD Webui Guide to running BroGAN'**](https://rentry.co/uoza8fqp), with a lot of info about BroGAN.
- Users at the **#men-only channel** on the [**Unstable Diffusion Discord server**](https://discord.gg/unstablediffusion), by who I was made aware of BroGAN (and other  *t a s t y*  #men-only generative-AI stuff 😏).
"""
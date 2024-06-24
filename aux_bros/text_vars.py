# Some text variables for the main script

title = "# 👨🏻👨🏼👨🏽👨🏾👨🏿 Adversarial Bros 👨🏿👨🏾👨🏽👨🏼👨🏻"

description = "A simple (yeah right) Gradio app that generates face photos of delicious (albeit imaginary) bros using BroGAN, a StyleGAN3 model."

open_output_folder_button_text = "📂 Open output folder"
save_single_image_button_text = "💾 Save the bro"

info_text = """
<p>🔵 There's two function modes at the moment:</p>
&emsp;&emsp;<u>Simple Image Generator:</u><br>
&emsp;&emsp;- Generates and displays (and optionally saves) a single image with a seed and truncation psi value of your choice.<br>
&emsp;&emsp;- Use mainly for experimenting with the model's different parameters, figure out how they affect the result, etc.<br>
&emsp;&emsp;<u>Bulk Generator:</u><br>
&emsp;&emsp;- Bulk generates and saves multiple images from various seeds and truncation psi values of your choice.<br>
<br>
<p>🔵 The generated images are in PNG format and have a size of 256x256 pixels.</p><br>
<p>🔵 The available parameters are (in ignorant layman's terms):</p>
&emsp;&emsp;<u>Seed:</u><br>
&emsp;&emsp;- Each seed is an integer and creates a unique face.<br>
&emsp;&emsp;- There's almost <i>4.3 billion</i> seeds available for your visual pleasure.<br>
&emsp;&emsp;<u>Truncation psi:</u><br>
&emsp;&emsp;- A decimal number ranging from 0.00 to 1.00.<br>
&emsp;&emsp;- Values further away from 0 create more diverse faces but with worse quality.<br>
&emsp;&emsp;- The recommended value from BroGAN's creator is 0.7, but I think I've been getting better results with 0.6 or 0.65.<br>
&emsp;&emsp;- Apparently, if the same psi value is negated, it gives a completely different image.<br>
<br>
<p>🔵 Check the credits section at the bottom of the page for more information about the model, its theoretical background and more.</p>
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
- [**NVlabs**](https://github.com/NVlabs/stylegan3) at NVIDIA, creators of the StyleGAN3 model that BroGAN is based on, and GitHub user [**PDillis**](https://github.com/PDillis) and their stylegan3 fork, [**stylegan3-fun**](https://github.com/PDillis/stylegan3-fun) whose code I've used as a backend here.
- GitHub user [**ZerxiesDerxies**](https://github.com/zerxiesderxies), author of the very helpful rentry.co guide [**'A men-only SD Webui Guide to running BroGAN'**](https://rentry.co/uoza8fqp), with a lot of info about BroGAN. And also author of the [**relevant A1111 extension**](https://github.com/zerxiesderxies/sd-webui-gan-generator/), which was the initial inspiration for this project.
- The devs of [**CodeFormer**](https://github.com/sczhou/CodeFormer), which is used to upscale the photos in this project.
- Users at the **#men-only channel** on the [**Unstable Diffusion Discord server**](https://discord.gg/unstablediffusion), by who I was made aware of BroGAN.
"""

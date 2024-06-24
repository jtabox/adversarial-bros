# A gradio app for image generation using the BroGAN StyleGAN3 PyTorch model - Main interface file

import os
import gradio as gr

from aux_bros.funcs import *
from aux_bros.text_vars import *

root_folder = os.path.dirname(__file__)
output_folder = "output"
model_folder = "model"
model_filename = "BroGANv1.0.0-upd-paths2-optimized.pkl"

with gr.Blocks(
    theme=gr.themes.Base(
        primary_hue=gr.themes.colors.blue,
        secondary_hue=gr.themes.colors.cyan,
        neutral_hue=gr.themes.colors.zinc,
        font=[gr.themes.GoogleFont("Roboto"), "serif"],
        font_mono=[gr.themes.GoogleFont("JetBrains Mono"), "monospace"],
        radius_size=gr.themes.sizes.radius_sm,
        spacing_size=gr.themes.sizes.spacing_sm,
        text_size=gr.themes.sizes.text_md,
    ),
    analytics_enabled=False,
    css="style.css",
) as main_ui:
    with gr.Group(visible=False):
        model_file = gr.Textbox(
            label="Model", value=os.path.join(root_folder, model_folder, model_filename)
        )
        output_folder = gr.Textbox(
            label="Output Folder", value=os.path.join(root_folder, output_folder)
        )
        single_image_filepath = gr.Textbox(label="Single Image Filepath", value="")
        last_gen_seed = gr.Textbox(label="Last Generation Seed", value="")
    gr.Markdown(value=title, elem_id="title")
    gr.Markdown(value=description, elem_id="description")
    with gr.Accordion(label="Information", elem_id="info-accordion", open=False):
        gr.HTML(value=info_text, elem_id="info-text")
    with gr.Tabs():
        with gr.Tab(label="Simple Image Generator", elem_id="simple-generator-tab"):
            with gr.Row():
                with gr.Column(scale=2):
                    with gr.Row():
                        simple_psi_slider = gr.Slider(
                            minimum=0,
                            maximum=1,
                            step=0.05,
                            value=0.65,
                            label="Truncation psi",
                            info="Values further away from 0 create more diverse faces but with worse quality. Recommended 0.6-0.7.",
                            interactive=True,
                            elem_id="simple-psi-slider",
                            scale=2,
                        )
                        simple_neg_psi_checkbox = gr.Checkbox(
                            value=False,
                            label="Negative psi",
                            info="Uses the negative of the psi value, which creates a completely different image than the positive one.",
                            interactive=True,
                            elem_id="simple-neg-psi-checkbox",
                            scale=1,
                        )
                    simple_seed_num = gr.Number(
                        label="Seed",
                        info="The seed used to generate the image. Can be between 0 and 4294967295, or -1 for a random seed.",
                        value=-1,
                        precision=0,
                        interactive=True,
                        minimum=-1,
                        maximum=4294967295,
                        step=1,
                        elem_id="simple-seed-number",
                    )
                with gr.Column(scale=1):
                    simple_seed_random_button = gr.Button(
                        value="🎲 Use random seed every run",
                        size="lg",
                        elem_id="simple-seed-random-button",
                        scale=1,
                    )
                    simple_seed_recycle_button = gr.Button(
                        value="♻️ Use the last run's seed",
                        size="lg",
                        elem_id="simple-seed-recycle-button",
                        scale=1,
                    )
            with gr.Row():
                with gr.Column():
                    simple_generate_button = gr.Button(
                        value="👨 Generate a bro 👨", variant="primary", size="lg"
                    )
                    simple_save_button = gr.Button(
                        value=save_single_image_button_text, variant="primary", size="sm"
                    )
                with gr.Column():
                    simple_result_image = gr.Image(
                        label="Result",
                        elem_id="simple-result-image",
                        format="png",
                        width=384,
                        height=384,
                        show_label=False,
                        type="numpy",
                        sources=None,
                        show_download_button=True,
                    )
                with gr.Column():
                    simple_output_text = gr.HTML(
                        label="Output",
                        value="<pre>Output will be shown here</pre>",
                        show_label=True,
                        visible=True,
                        elem_classes="output-text",
                    )
            simple_seed_random_button.click(
                fn=set_gen_seed,
                show_progress=False,
                inputs=[],
                outputs=[simple_seed_num],
            )
            simple_seed_recycle_button.click(
                fn=set_gen_seed,
                show_progress=False,
                inputs=[last_gen_seed],
                outputs=[simple_seed_num],
            )
            simple_generate_button.click(
                fn=generate_single_image,
                show_progress=False,
                inputs=[
                    model_file,
                    simple_seed_num,
                    simple_psi_slider,
                    simple_neg_psi_checkbox,
                ],
                outputs=[
                    simple_result_image,
                    last_gen_seed,
                    single_image_filepath,
                    simple_output_text,
                    simple_save_button,
                ],
            )
            simple_save_button.click(
                fn=save_single_image,
                show_progress=False,
                inputs=[
                    simple_result_image,
                    output_folder,
                    single_image_filepath,
                    simple_output_text,
                ],
                outputs=[simple_output_text],
            )
        with gr.Tab(label="Bulk Image Generator", elem_id="bulk-generator-tab"):
            with gr.Accordion(
                label="How to use", elem_id="bulk-usage-info-accordion", open=False
            ):
                gr.HTML(value=bulk_info_text, elem_id="bulk-info-text")
            with gr.Row():
                bulk_amount_textbox = gr.Number(
                    label="1. Amount of images to generate randomly",
                    info="Can be 0, blank or a positive integer.",
                    value=10,
                    precision=0,
                    interactive=True,
                    minimum=0,
                    step=1,
                    elem_id="bulk-amount-number",
                )
                bulk_seed_textbox = gr.Textbox(
                    label="2. Seeds to generate images from",
                    info="Can be a list of integers/ranges of integers between 0 and 4294967295, blank or -1.",
                    value="0-2, 20, 35-37",
                    placeholder="0-2, 20, 35-37",
                    lines=1,
                    max_lines=1,
                    elem_id="bulk-seeds-textbox",
                    interactive=True,
                )
                bulk_psi_values = gr.Textbox(
                    label="3. Truncation psi values to use",
                    info="Can be a list of decimals from -1.00 to 1.00 or blank.",
                    value="-0.65, 0.7",
                    placeholder="-0.65, 0.7",
                    lines=1,
                    max_lines=1,
                    elem_id="bulk-psi-values",
                    interactive=True,
                )
            with gr.Row():
                bulk_generate_button = gr.Button(
                    value="👬👬 Generate many bros 👬👬", variant="primary"
                )
            with gr.Row():
                bulk_output_text = gr.HTML(
                    label="Output",
                    value="<pre>Output will be shown here</pre>",
                    show_label=True,
                    visible=True,
                    elem_classes="output-text",
                )
            with gr.Row():
                bulk_save_destination = gr.Textbox(
                    label="Output folder",
                    value=output_folder.value,
                    info="Folder to save the images in. Will be created if it doesn't exist.",
                    placeholder="output",
                    lines=1,
                    max_lines=1,
                    elem_id="bulk-save-destination",
                    interactive=True,
                )
                bulk_save_destination_button = gr.Button(
                    value="📁 Set output folder", variant="secondary"
                )
                open_output_folder_button = gr.Button(
                    value=f"{open_output_folder_button_text} ({output_folder.value})",
                    variant="secondary",
                    size="sm",
                    elem_id="open-output-folder-button",
                )
            bulk_generate_button.click(
                fn=bulk_generate_images,
                inputs=[
                    model_file,
                    bulk_seed_textbox,
                    bulk_psi_values,
                    bulk_amount_textbox,
                    bulk_save_destination,
                ],
                outputs=[bulk_output_text],
            )
            bulk_amount_textbox.change(
                fn=bulk_update_amount,
                inputs=[bulk_amount_textbox, bulk_seed_textbox, bulk_psi_values],
                outputs=[bulk_generate_button, bulk_output_text],
            )
            bulk_seed_textbox.change(
                fn=bulk_update_amount,
                inputs=[bulk_amount_textbox, bulk_seed_textbox, bulk_psi_values],
                outputs=[bulk_generate_button, bulk_output_text],
            )
            bulk_psi_values.change(
                fn=bulk_update_amount,
                inputs=[bulk_amount_textbox, bulk_seed_textbox, bulk_psi_values],
                outputs=[bulk_generate_button, bulk_output_text],
            )
            bulk_save_destination_button.click(
                fn=set_output_folder,
                inputs=bulk_save_destination,
                outputs=[output_folder, open_output_folder_button],
            )
            open_output_folder_button.click(
                fn=open_output_folder, inputs=output_folder, outputs=[]
            )
    with gr.Accordion(label="Credits", elem_id="credits-accordion", open=False):
        gr.Markdown(value=credits_text, elem_id="credits-text")

if __name__ == "__main__":
    main_ui.launch(inbrowser=True)

# A gradio app for image generation using the BroGAN StyleGAN3 PyTorch model - Main interface file

import os
import gradio as gr

from funcs import use_last_gen_seed, generate_image
from text_vars import title, description, info_text, credits_text


output_folder = os.path.join(os.path.dirname(__file__), "output")
model_folder = os.path.join(os.path.dirname(__file__), "model")
model_filename = "BroGANv1.0.0.pkl"

with gr.Blocks(analytics_enabled=False, css='style.css') as main_ui:
    with gr.Group(visible=False):
        model_file = gr.Textbox(label="Model", value=os.path.join(model_folder, model_filename))
    gr.Markdown(value=title, elem_id="title")
    gr.Markdown(value=description, elem_id="description")
    with gr.Accordion(label="Information", elem_id="info-accordion", open=False):
        gr.HTML(value=info_text, elem_id="info-text")
    with gr.Tabs():
        with gr.Tab(label="Simple Image Generator", elem_id="simple-generator-tab"):
            with gr.Row():
                with gr.Column(scale=2):
                    psi_slider = gr.Slider(minimum=-1, maximum=1, step=0.05, value=0.7, label="Truncation psi (recommended 0.7)", interactive=True, elem_id="psi-slider")
                    seed_num = gr.Number(label="Seed (0 - 4294967295)", value=-1, precision=0, interactive=True, minimum=-1, maximum=4294967295, step=1, elem_id="seed-num")
                with gr.Column(scale=1):
                    seed_random_button = gr.Button(value="🎲 Use random seed every run", size="lg", elem_id="seed-random-button", scale=1)
                    seed_recycle_button = gr.Button(value="♻️ Use the last run's seed", size="lg", elem_id="seed-recycle-button", scale=1)
            with gr.Row():
                with gr.Column():
                    run_simple_gen_button = gr.Button(value="Generate Simple Image", variant="primary")
                with gr.Column():
                    result_image = gr.Image(label="Result", elem_id="result-image", format="png", width=384, height=384)
                with gr.Column():
                    seed_text = gr.Markdown(label="Output Seed", elem_id="seed-text")

            seed_random_button.click(fn=lambda: -1, show_progress=False, inputs=[], outputs=seed_num)
            seed_recycle_button.click(fn=use_last_gen_seed, show_progress=False, inputs=[seed_text], outputs=[seed_num])
            run_simple_gen_button.click(fn=generate_image, show_progress=False, inputs=[model_file, seed_num, psi_slider], outputs=[result_image, seed_text])
        with gr.Tab(label="Bulk Generator", elem_id="bulk-generator-tab"):
            gr.Markdown(value="## NYI: Bulk Generator")
            with gr.Row():
                open_output_folder_button = gr.Button(value=f"📁 Open output folder ({output_folder})", size="lg", elem_id="open-output-folder-button")
                open_output_folder_button.click(lambda: os.startfile(output_folder))
    with gr.Accordion(label="Credits", elem_id="credits-accordion", open=False):
        gr.Markdown(value=credits_text, elem_id="credits-text")


if __name__ == "__main__":
    main_ui.launch()

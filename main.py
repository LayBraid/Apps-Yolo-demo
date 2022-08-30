from transformers import AutoFeatureExtractor, YolosForObjectDetection
import gradio as gr
from PIL import Image
import torch
import matplotlib.pyplot as plt
import io

COLORS = [[0.000, 0.447, 0.741], [0.850, 0.325, 0.098], [0.929, 0.694, 0.125],
          [0.494, 0.184, 0.556], [0.466, 0.674, 0.188], [0.301, 0.745, 0.933]]

pre_process = AutoFeatureExtractor.from_pretrained(f"hustvl/yolos-base")
forward = YolosForObjectDetection.from_pretrained(f"hustvl/yolos-base")


def infer(img, prob_threshold: int):

    img = Image.fromarray(img)

    pixels = pre_process(img, return_tensors="pt").pixel_values

    with torch.no_grad():
        outputs = forward(pixels, output_attentions=True)

    probas = outputs.logits.softmax(-1)[0, :, :-1]
    keep = probas.max(-1).values > prob_threshold

    target_sizes = torch.tensor(img.size[::-1]).unsqueeze(0)
    postprocessed_outputs = pre_process.post_process(outputs, target_sizes)
    bboxes_scaled = postprocessed_outputs[0]['boxes']

    res_img = plot_results(img, probas[keep], bboxes_scaled[keep], forward)

    return res_img


def plot_results(pil_img, prob, boxes, model):
    plt.figure(figsize=(16, 10))
    plt.imshow(pil_img)
    ax = plt.gca()
    colors = COLORS * 100
    for p, (xmin, ymin, xmax, ymax), c in zip(prob, boxes.tolist(), colors):
        cl = p.argmax()
        object_class = model.config.id2label[cl.item()]

        ax.add_patch(plt.Rectangle((xmin, ymin), xmax - xmin, ymax - ymin,
                                   fill=False, color=c, linewidth=3))
        text = f'{object_class}: {p[cl]:0.2f}'
        ax.text(xmin, ymin, text, fontsize=15,
                bbox=dict(facecolor='yellow', alpha=0.5))
    plt.axis('off')
    return fig2img(plt.gcf())


def fig2img(fig):
    buf = io.BytesIO()
    fig.savefig(buf)
    buf.seek(0)
    img = Image.open(buf)
    return img

image_in = gr.components.Image()
image_out = gr.components.Image()
prob_threshold_slider = gr.components.Slider(minimum=0, maximum=1.0, step=0.05, value=0.9,
                                             label="Probability Threshold")

Iface = gr.Interface(
    fn=infer,
    inputs=[image_in, prob_threshold_slider],
    outputs=image_out,
    title="LayBraid © 2022 | Object Detection",
).launch()

# DepthCrafter Nodes

**Create consistent depth maps for your videos using DepthCrafter in ComfyUI.**

Original DepthCrafter repo: https://github.com/Tencent/DepthCrafter

## ⭐ Example Workflow:
<img src="https://i.imgur.com/gtL91SR.png" alt="Description" width="800"/>

## 📦 Included Nodes:
- **DownloadAndLoadDepthCrafterModel**: Will fetch the model files need to run DepthCrafter and save them under models/depthcrafter.
- **DepthCrafter**: Renders out depthmap videos given the following inputs:
  - **depthcrafter_model**: (input from the first node)
  - **images**: (single or batch),
  - **max_res**: the maximum resolution of the input images, supports increments of 64 pixels. (Larger resolutions require more VRAM)
  - **max_inference_steps**: more steps may result in less artifacts in the output, but will take longer to render.
  - **guidance_scale**: (1 - 1.2 recommended)
  - **window_size**: the length of the context window for DepthCrafter. You can lower this to save on VRAM at the expense of taking longer to render (75-110 recommended)
  - **overlap**: how much to overlap each context window to render longer videos > 110 frames. (25 recommended)

## 🔧 Installation and Usage

1. ComfyUI Manager:

- This node pack is available to install via the [ComfyUI Manager](https://github.com/ltdrdata/ComfyUI-Manager). You can find it in the Custom Nodes section by searching for "DepthCrafter" and clicking on the entry called "DepthCrafter Nodes".

2. Clone the repository:
- Navigate to ComfyUI/custom_nodes folder in terminal or command prompt.
- Clone the repo using the following command:
```bash
git clone https://github.com/akatz-ai/ComfyUI-DepthCrafter-Nodes.git
```
- Restart ComfyUI

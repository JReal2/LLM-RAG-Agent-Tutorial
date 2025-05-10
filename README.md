# LLM, RAG and AI Agent Tutorial
This repository contains [Syllabus](https://github.com/mac999/LLM-RAG-Agent-Tutorial/blob/main/1-1.prepare/syllabus-llm-rag-agent.docx) including AI tools installation and documents for LLM, RAG, AI Agent and MCP development, focusing on creative LLM coding, modeling, and computing as the viewpoint of project development. The colab code, source, presentation like below can be used for developing LLM, RAG and AI Agent. 
- [Transformer encoder and decoder tutoring](https://github.com/mac999/LLM-RAG-Agent-Tutorial/tree/main/1-2.transformer). [Transformer scratch source code](https://github.com/mac999/LLM-RAG-Agent-Tutorial/tree/main/2-3.deep-tranformer)
- Token and Embedding for NLP (natural language process) using [huggingface](https://github.com/mac999/LLM-RAG-Agent-Tutorial/tree/main/2-1.huggingface)
- [Multi-modal](https://github.com/mac999/LLM-RAG-Agent-Tutorial/tree/main/3-3.multi-modal) like CLIP, LLaVA
- Stable Diffusion and prompt engineering using [text to image, video, audio, sound, document (word, presentation) and code (app, game) tools](https://github.com/mac999/LLM-RAG-Agent-Tutorial/tree/main/2-2.genai-prompt) 
- LLM. [Train and Finetune](https://github.com/mac999/LLM-RAG-Agent-Tutorial/tree/main/3-2.finetuning) for model like gemma, llama
- [RAG](https://github.com/mac999/LLM-RAG-Agent-Tutorial/tree/main/4-2.llm-rag) and [Langchain](https://github.com/mac999/LLM-RAG-Agent-Tutorial/tree/main/4-1.langchain). [Vector DB](https://github.com/mac999/LLM-RAG-Agent-Tutorial/tree/main/4-3.db) like [FAISS](https://engineering.fb.com/2017/03/29/data-infrastructure/faiss-a-library-for-efficient-similarity-search/), Chroma DB, Graph DB using Neo4j
- [Chatbot](https://github.com/mac999/LLM-RAG-Agent-Tutorial/tree/main/4-3.db) with [Ollama](https://github.com/mac999/LLM-RAG-Agent-Tutorial/tree/main/4-5.ollama). Gradio and Streamlit for UX
- [AI Agent](https://github.com/mac999/LLM-RAG-Agent-Tutorial/tree/main/5-1.agent) and [MCP(Model Context Protocol)](https://github.com/mac999/LLM-RAG-Agent-Tutorial/tree/main/5-2.llm-mcp-app)
- LLM Internal Code Analysis like [Deepseek](https://github.com/mac999/LLM-RAG-Agent-Tutorial/tree/main/2-4.deep-seek), Manus</br>
- [Vibe coding](https://github.com/mac999/LLM-RAG-Agent-Tutorial/tree/main/5-3.vibe-coding) using Copilot, GPT etc

<img src="https://github.com/mac999/BIM_LLM_code_agent/raw/main/doc/img1.gif" height="300"/><img src="https://github.com/mac999/geo-llm-agent-dashboard/raw/main/doc/geo_llm_demo.gif" height="300"/>

# Preparation for LLM, RAG and AI Agent study
LLM uses deep learning model architecture like transformer which uses numerical analysis, linear algebra, so it's better to understand the below subjects before starting it.
- [linear algebra](https://github.com/mac999/LLM-RAG-Agent-Tutorial/blob/main/1-1.prepare/linear-algebra.pdf)
- [newton mathod for equation solution](https://www.intmath.com/applications-differentiation/newtons-method-interactive.php). In addition, [differential Calculus](https://www.geogebra.org/t/differential-calculus) includes newton method.
<img src="https://github.com/mac999/LLM-RAG-Agent-Tutorial/blob/main/1-1.prepare/numerical-analysis-newton.PNG" alt="newton method" width="400"/>

- [numerical analysis(Youtube)](https://www.youtube.com/watch?v=bfoxcZYoGfQ)
- [numerical analysis reference](https://github.com/mac999/LLM-RAG-Agent-Tutorial/blob/main/1-1.prepare/numerical-analysis.pdf)

# Installation for LLM, RAG and AI agent development
First, clone this repository. 
```bash
git clone https://github.com/mac999/LLM-RAG-Agent-Tutorial.git
```
Second, check [syllabus](https://github.com/mac999/LLM-RAG-Agent-Tutorial/blob/main/1-1.prepare/syllabus-llm-rag-agent.docx) to understand LLM, RAG and AI agent development course.</br>
<img src="https://github.com/mac999/LLM-RAG-Agent-Tutorial/blob/main/1-2.transformer/transformer-architecture.PNG?raw=true" height="300" /></br>
Before running the example code, ensure you have Colab Pro, Python 3.10 or higher installed. Some tool or library use NVIDIA GPU, so if you want to use it, prepare notebook computer with NVIDIA GPU(recommend 8GB. minimum 4GB)
Follow the instructions below to set up your environment:
- [LLM development environment document(word file)](https://github.com/mac999/LLM-RAG-Agent-Tutorial/blob/main/1-1.prepare/dev-env.docx)

In refernce, this lesson will use the below 
- **OpenAI**: To use ChatGPT LLM model, You need to create OpenAI API key
- **Huggingface**: For uisng LLM, Stable Diffusion-based model, You need to sign up Huggingface. In example, [Single Image-to-3D model](https://huggingface.co/spaces/stabilityai/stable-point-aware-3d)
- **Ollama**: For using AI tools in interactive art projects. You need to install NVIDIA cuda for run it.

## NVIDIA Drivers (for Ollama. optional)
For GPU-accelerated tasks, you need to install the correct NVIDIA drivers for your GPU.

- **Download NVIDIA Drivers**: [NVIDIA Driver Downloads](https://www.nvidia.com/Download/index.aspx)
- **Steps**:
  1. Identify your GPU model using the NVIDIA Control Panel or the `nvidia-smi` command in the terminal.
  2. Download the latest driver for your GPU model.
  3. Install the driver and reboot your system.

To confirm installation:
```bash
nvidia-smi
```

## CUDA Toolkit (for NVIDIA. optional)
CUDA is required for running GPU-accelerated operations.

- **Download CUDA Toolkit**: [CUDA Toolkit](https://developer.nvidia.com/cuda-downloads)
- During installation:
  - Match the CUDA version with your NVIDIA driver and GPU model. Refer to the compatibility chart on the download page.
  - Install the CUDA Toolkit with default options.
  - Add the CUDA binary paths to your environment variables.

## Python
Ensure that Python (safe version 3.10 or 3.12) is installed on your system. About macbook, please refer to [how to install python on mac](https://www.youtube.com/watch?v=u4xUUBTER4I).

- **Download Python**: [python.org](https://www.python.org/)
- During installation:
  - Check the box for **"Add Python to PATH"**.
  - Install Python along with **pip**.

To confirm installation in terminal(DOS command in windows. Shell terminal in linux):
```bash
python --version
```

## Anaconda
Ensure that Anaconda (version 24.0 or later) is installed on your system.

- **Download Anaconda**: [Anaconda](https://docs.anaconda.com/anaconda/install/)

### Huggingface, OpenAI (Optinnal) Account 
Make Accounts for OpenAI, Huggingface
- Sign up [Huggingface](https://huggingface.co/) and make [API token](https://huggingface.co/settings/tokens)to develop Open source LLM-base application
- Sign up [OpenAI API](https://platform.openai.com/) to develop ChatGPT-based application (*Note: don't check auto-subscription option)

## PyTorch library 

- If AI-related models or tools will be used (such as LLM model fine-tuning with Ollama), install stable [PyTorch](https://pytorch.org/get-started/locally/)(11.8 version) and additional packages:
   ```bash
   pip install openai
   pip install torch torchvision torchaudio --index-url https://download.pytorch.org/whl/cu118
   ```

## Install Python Packages
Run the following command to install the required libraries:
```bash
pip install pandas numpy
pip install ollama openai transformers huggingface_hub langchain
```

## Install Ollama 
For examples that utilize Ollama, follow the installation instructions from the [Ollama website](https://www.ollama.com/).

## Blender (for AI-Assisted Modeling)
If the script or application involves Blender for 3D modeling, ensure Blender is installed.

- **Download Blender**: [blender.org](https://www.blender.org/download/)
- After installation:
  - Enable the **Python Console** within Blender to run scripts directly.
  - Ensure Blender uses the same Python environment where required libraries are installed.

## Install sublime and vscode (Recommend)
Install [Sublime](https://www.sublimetext.com/) for editing source code</br>
Install [vscode](https://code.visualstudio.com/download) for debuging code. Please refer to how to [install vscode](https://www.youtube.com/watch?v=vesxpfOAOCw).</br>

---

# **System Environment Checks**

After completing the installations, verify that the environment is set up correctly:

1. **Check Python Version**:
   ```bash
   python --version
   ```

2. **Verify NVIDIA Drivers**:
   ```bash
   nvidia-smi
   ```

3. **Confirm CUDA Version**:
   ```bash
   nvcc --version
   ```

4. **Test Python Libraries**:
   Create a test script and import the installed libraries:
   ```python
   import p5 # only, python 3.10, working
   import pandas as pd
   import numpy as np
   print("Libraries are installed successfully!")
   ```

# For media art
If you're interested in media art, refer to the below link. The repository includes examples to experiment with generative media art.</br>
- [Gen AI for Media Art](https://github.com/mac999/llm-media-art-demo)
<img src="https://github.com/mac999/blender-llm-addin/raw/main/doc/blender_gpt.gif" height="300"/>

In addition, you can find Text-to-3D model tool the below link. 
- [Text-to-3D model code](https://github.com/mac999/blender-llm-addin): Using Open-Source Models with Blender for AI-Assisted 3D Modeling: Comparative Study with OpenAI GPT

# Reference
- [NVIDIA cuda programming, open source and AI](https://www.slideshare.net/slideshow/nvidia-cuda-programming-open-source-and-ai/270372806?from_search=6)
- [Chat with ChatGPT through Arduino IoT Cloud](https://projecthub.arduino.cc/dbeamonte_arduino/chat-with-chatgpt-through-arduino-iot-cloud-6b4ef0)
- [chatGPT-Arduino-library](https://github.com/programming-electronics-academy/chatGPT-Arduino-library/tree/main)
- [ChatGPT_Client_For_Arduino](https://github.com/0015/ChatGPT_Client_For_Arduino)
- [I tried ChatGPT for Arduino - It’s Surprising](https://blog.wokwi.com/learn-arduino-using-ai-chatgpt/)
- [Using ChatGPT to Write Code for Arduino and ESP32](https://dronebotworkshop.com/chatgpt/)

# License
This repository is licensed under the MIT License. You are free to use, modify, and distribute the code for personal or commercial projects.

# Author
laputa99999@gmail.com

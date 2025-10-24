# ModelLite

## Abstract

This repository implements a comprehensive **Post-Training Quantization (PTQ) and model export pipeline** for PyTorch models. Designed with a universal, modular architecture, it provides an end-to-end solution from model fine-tuning to quantized deployment.

**Current Implementation**: The pipeline is demonstrated with MobileNetV3, serving as a reference implementation that showcases the framework's capabilities for computer vision tasks.

**Future Vision**: The architecture is model-agnostic by design, with plans to extend support to additional model families and enhance quantization strategies. This project bridges the gap between research experimentation and production deployment, offering researchers and engineers a robust foundation for model optimization workflows.

**Note**: Read [TODO List](#todo-list) for more details of future version.

## Vision & Goals

### **Universal Model Optimization**

Create a single, config-driven pipeline that works across any PyTorch model architecture, eliminating the need for model-specific optimization code.

### **Streamlined Production Pipeline**

Bridge the gap between research and deployment with an end-to-end workflow: fine-tuning → quantization (PTQ-focused) → export → deployment analysis.

## Install

```shell
git clone <repository-url>
cd ModelLite
conda create -n model_lite python=3.12
conda activate model_lite
pip install -r requirements.txt
```

Run this command to install PyTorch 2.7.1+cu118:

```shell
pip install torch torchvision torchaudio --index-url https://download.pytorch.org/whl/cu118
```

**Note**: The reason to use `cu118` is just limited support of my GPU `GeForce MX350`.

## Quick Start

```shell
# Default mode.
python main.py

# Debug mode.
python main.py run=debug

# Edit configuration.
python main.py data.batch_size=32 training.num_epochs=10
```

> **Later this will be updated...**

## Features

### **Core Capabilities**

- **Model-Agnostic Framework** - Works seamlessly with any PyTorch model architecture
- **Production-Ready Quantization** - Battle-tested techniques with comprehensive validation
- **Research-Friendly Design** - Modular architecture for easy experimentation and extension

### **Quantization Excellence**

- **Multi-Strategy Quantization** - Support for FX Graph, Static, and Dynamic quantization
- **Zero-Boilerplate Integration** - Minimal code changes required for existing models
- **Configurable Precision** - Flexible quantization presets for different deployment scenarios

### **Deployment Ready**

- **Universal Model Export** - One-click export to ONNX, TensorRT, and other inference engines
- **Embedded Deployment Analysis** - Comprehensive compatibility checking for edge devices
- **Performance Benchmarking** - Detailed speed, accuracy, and memory usage analysis

### **Engineering Excellence**

- **End-to-End Pipeline** - Unified workflow from fine-tuning to quantization and deployment
- **Modular & Configurable** - Clean separation with Hydra-based configuration management
- **Production Monitoring** - Built-in validation, debugging, and performance tracking

### **Unique Value Propositions**

- **Smart Deployment Analysis** - Automatic compatibility checking for Raspberry Pi, Jetson, and other edge devices
- **Quantization Health Checks** - Comprehensive validation to ensure quantization effectiveness
- **Performance Trade-off Analysis** - Clear insights into accuracy vs. speed vs. size trade-offs
- **One-Click Workflows** - From trained model to deployed artifact in a single command

## Dependencies

### Basic

```text
torch>=1.8.0
torchvision>=0.9.0
hydra-core>=1.3.0
omegaconf>=2.3.0
tqdm>=4.64.0
```

### Dev

```text
torch==2.0.1
torchvision==0.15.2
hydra-core==1.3.2
omegaconf==2.3.0
tqdm==4.65.0
numpy==1.24.3
Pillow==9.5.0
tensorboard==2.13.0
matplotlib==3.7.1
```

## TODO List

- [ ] Improve save_model (by timestamp).
- [ ] Implement logging.
- [ ] Implement Exception Handling.
- [ ] Validate Hydra Config.
- [ ] Implement various model support.
- [ ] Implement various dataset support.
- [ ] Implement more benchmark to evaluate model.
- [ ] Compile the model.
- [ ] Implement device controller.
- [ ] Implement Model Pruning.
- [ ] Implement Model Distilling.
- [ ] Introduce the framework like AutoML.
- [ ] Make more quantization presets.
- [ ] Check cached processing results in load_dataset.
        ```python
        def load_dataset(cfg: DatasetContextConfig):
            cache_dir = Path(cfg.root_dir) / "cached"
            if cache_dir.exists():
                pass
        ```
- [ ] Implement dataloader parallel.
- [ ] Check model status in each process.
- [ ] Beautify terminal outputs.

## Project Structure

```shell
MobileNet
├── analysis
│   ├── benchmark.py
│   ├── complexity.py
│   ├── deployment.py
│   ├── __init__.py
│   ├── quantization_test.py
│   └── utils.py
├── dataloader.py
├── Dataset
├── debug.py
├── enumeration.py
├── export
│   ├── decorator.py
│   ├── engine.py
│   └── __init__.py
├── hydra_configs
│   ├── config.yaml
│   ├── data
│   │   └── cifar10.yaml
│   ├── evaluation
│   │   └── default.yaml
│   ├── export
│   │   ├── common.yaml
│   │   └── engine
│   │       └── onnx.yaml
│   ├── model
│   │   └── mobilenet.yaml
│   ├── quantization
│   │   └── fx_graph.yaml
│   ├── run
│   │   ├── debug.yaml
│   │   └── default.yaml
│   └── training
│       ├── default.yaml
│       ├── optimizer
│       │   └── adam.yaml
│       └── scheduler
│           └── cosine.yaml
├── logs
├── main.py
├── model
│   ├── modify.py
│   ├── train.py
│   └── utils.py
├── outputs
├── quantization
│   ├── __init__.py
│   ├── qconfig_presets.py
│   ├── quantize_functions.py
│   └── quantize.py
├── README.md
├── requirements.txt
├── saved_models
└── schemas
    ├── context_classes.py
    ├── data_classes.py
    ├── functions.py
    ├── __init__.py
    ├── presets.py
    └── validation.py

21 directories, 40 files
```

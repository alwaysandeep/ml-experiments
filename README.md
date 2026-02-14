# ML Experiments

PyTorch and FastAI experimentation environment for Jupyter notebooks.

## Setup

### Prerequisites
- Python 3.11+
- pip

### Installation

1. **Install uv package manager**
   ```bash
   pip install uv
   ```

2. **Create virtual environment**
   ```bash
   uv venv .venv
   ```

3. **Install PyTorch (CPU-only)**
   ```bash
   # Windows
   uv pip install --python .venv/Scripts/python.exe torch torchvision torchaudio --index-url https://download.pytorch.org/whl/cpu

   # Linux/Mac
   uv pip install --python .venv/bin/python torch torchvision torchaudio --index-url https://download.pytorch.org/whl/cpu
   ```

4. **Install other dependencies**
   ```bash
   # Windows
   uv pip install --python .venv/Scripts/python.exe -r requirements.txt

   # Linux/Mac
   uv pip install --python .venv/bin/python -r requirements.txt
   ```

5. **Register Jupyter kernel**
   ```bash
   # Windows
   .venv/Scripts/python.exe -m ipykernel install --user --name ml-experiments --display-name "ML Experiments"

   # Linux/Mac
   .venv/bin/python -m ipykernel install --user --name ml-experiments --display-name "ML Experiments"
   ```

6. **Select kernel in VS Code**
   - Open any `.ipynb` file in the `experiments/` folder
   - Click the kernel picker (top-right)
   - Select **"ML Experiments"**
   - If not visible, reload VS Code: `Ctrl+Shift+P` → `Developer: Reload Window`

## Verify Installation

Run this in a notebook cell:
```python
import torch
import fastai
print(f"PyTorch: {torch.__version__}")
print(f"FastAI: {fastai.__version__}")
```

## Environment Info
- **Python**: 3.11.9
- **PyTorch**: 2.10.0+cpu (CPU-only)
- **FastAI**: 2.8.6
- **Package Manager**: uv

# Enhanced Badminton Player Pose Estimation Using Improved YOLOv8-Pose with Efficient Local Attention

## Quick Start

### Installation

1. **Install PyTorch**:
   Ensure you have PyTorch installed. You can install it based on your system configuration by following the [PyTorch official website](https://pytorch.org/get-started/locally/).

2. **Install Dependencies**:
   ```bash
   pip install -r requirements.txt
   ```
3. **Create Necessary Directories**:
   Initialize directories for training dataset:
   ```bash
   mkdir dataset
   ```

### Project Structure
Your project directory should be structured as follows:
```plaintext
${PROJECT_ROOT}
├── dataset
├── blocks
├── ultraytics
├── yamlfile
├── posetrain.py
├── README.md
├── requirements.txt
```

---

## Data Preparation

1. **Download Dataset**:
   You can download https://pan.baidu.com/s/1ECD7hMsq-mu_32PjysJonA?pwd=0000 提取码: 0000

## Training

To train your model, run the following command:
```bash
python posetrain.py
```


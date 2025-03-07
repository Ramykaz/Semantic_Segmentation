# Image Segmentation using HRNet

This project provides a segmentation pipeline using a pretrained HRNet model from the **LoveDA** dataset. The model performs semantic segmentation on input satellite images and outputs a segmentation mask.

## Setup Instructions

### 1. Clone the Repository

Clone the repository to your local machine:

```bash
git clone https://github.com/your-username/your-repo.git
cd your-repo
2. Install Dependencies
Create a virtual environment (optional but recommended) and install the dependencies listed in requirements.txt.

bash
Copy
Edit
python -m venv venv
source venv/bin/activate  # For Windows: venv\Scripts\activate
pip install -r requirements.txt
3. Set Up LoveDA Dataset
This repository relies on the LoveDA dataset, which includes the model checkpoint (hrnetw32.pth) and configuration files. You will need to set up LoveDA to download and use these files.

Steps to Set Up LoveDA:
Clone the LoveDA repository from LoveDA GitHub.

bash
Copy
Edit
git clone https://github.com/Project-LoveDA/loveda.git
Navigate to the LoveDA/Semantic_Segmentation/configs/baseline/ directory.

Download or ensure that the hrnetw32.pth checkpoint and hrnetw32.py config file are present.

Update the paths in the main.py script to point to the LoveDA directory's checkpoint and config file:

python
Copy
Edit
ckpt_path = "/path/to/LoveDA/Semantic_Segmentation/configs/baseline/hrnetw32.pth"  # Adjust path
config_path = "/path/to/LoveDA/Semantic_Segmentation/configs/baseline/hrnetw32.py"  # Adjust path
4. Run the Script
Once you have set up the dependencies and LoveDA, you can run the segmentation script by providing the image path as an argument.

bash
Copy
Edit
python main.py --image-path /path/to/your/image.tif
This will generate a segmentation mask and save it in the same location as the input image.

Example:
bash
Copy
Edit
python main.py --image-path /content/drive/MyDrive/image.tif
5. Output
After running the script, the segmentation mask will be saved in the same directory as the input image. The output will have the same dimensions as the input image, and the mask will be color-coded based on the segmentation classes:

0: Background (Black)
1: Building (Dark Red)
2: Road (Gray)
3: Water (Blue)
4: Barren (Yellow)
5: Forest (Green)
6: Agriculture (Light Yellow)
For example, if your input image is image.tif, the output will be saved as image_mask.tif.

Notes
LoveDA Dataset: Make sure you have the LoveDA model files correctly downloaded. The paths to the checkpoint file (hrnetw32.pth) and config file (hrnetw32.py) need to be set correctly in the main.py script.
Image Preprocessing: The script resizes images to 512x512 for segmentation and then resizes the mask to the original image dimensions.
TensorFlow or PyTorch: The model is implemented in PyTorch, so you need to have PyTorch installed in your environment.
License
This project is licensed under the MIT License - see the LICENSE file for details.

Acknowledgements
LoveDA Dataset: The LoveDA dataset is used for training and evaluating the model. Please refer to their GitHub repository for more details.
HRNet: The HRNet model is based on the work by HRNet: High-Resolution Representations for Visual Recognition.
pgsql
Copy
Edit

This is a complete and formatted `README.md` file for your project. You can place it in your project directory and update any necessary links or paths to sui

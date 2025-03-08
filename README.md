# Image Segmentation using HRNet

This project provides a segmentation pipeline using a pretrained HRNet model from the LoveDA dataset. The model performs semantic segmentation on input satellite images and outputs a segmentation mask.

## Setup Instructions

### 1. Clone the Repository
Clone the repository to your local machine:

```bash
git clone https://github.com/your-username/your-repo.git
cd your-repo

2. Install Dependencies
Before running the project, install the required dependencies:
pip install -r requirements.txt

3. Prepare the Dataset
Download the LoveDA dataset and place it in the data/ directory. The dataset should include both training and testing images.

4. Run the Segmentation
Run the segmentation model using the following command:

python segment.py --input <path-to-input-image> --output <path-to-output-mask>
Replace <path-to-input-image> with the path to the input satellite image and <path-to-output-mask> with the desired output path for the segmentation mask.

5. View Results
Once the segmentation is completed, you can visualize the output mask using any image viewer.

Model Details
The segmentation is based on the HRNet (High-Resolution Network) model, which is pretrained on the LoveDA dataset. It is designed to provide high-quality segmentation results, particularly for satellite imagery.

License
This project is licensed under the MIT License - see the LICENSE file for details.

This `README.md` file provides a clear and structured guide for setting up and using the image segmentation project based on the HRNet model. It includes sections for cloning the repository, installing dependencies, preparing the dataset, running the segmentation, viewing results, and details about the model and license.

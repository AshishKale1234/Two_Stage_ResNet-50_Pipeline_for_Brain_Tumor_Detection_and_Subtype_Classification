# Two-Stage ResNet50 Pipeline for Brain Tumor Detection and Subtype Classification

---

## 1. Dataset

### 1.1 This project uses the **Brain Tumor MRI Dataset** from Kaggle, which can be downloaded from the below URL

- Kaggle URL: <https://www.kaggle.com/datasets/akrashnoor/brain-tumor>

Because Kaggle requires an authenticated account and API token, the dataset is **not automatically downloaded by this code**.

### 1.2 A zip file will be downloaded and please extract it to the project folder

The original dataset is organized into two high-level classes:

- `2 classes/` – non–brain-tumor MRIs
  -  `yes/`
  -  `no/`
- `4 classes/` – MRIs with a tumor present, further split into:
  - `glioma_tumor/`
  - `meningioma_tumor/`
  - `normal/`
  - `pituitary_tumor/`  

---

## 2. Create an active virtual environment

It is recommended to run this project inside a Python virtual environment so that all dependencies are isolated from your global Python installation.
 
### 2.1 Pre-requisites

- Python 3.10+ installed (check with `python --version` or `python3 --version`)
- `pip` installed (usually comes with Python)

### 2.2 Create a virtual environment

From the root of the project (where `requirements.txt` lives), run:

**for macOS and Linux:**
```bash
python3 -m venv venv
```

**for Windows (PowerShell or CMD):**
```bash
python -m venv venv
```

### 2.3 Activate the virtual environment

**for Windows(Powershell)**
```bash
venv\Scripts\Activate.ps1
```

**for Windows(CMD)**
```bash
venv\Scripts\activate.bat
```

**for macOS/Linux**
```bash
source venv/bin/activate
```

### 2.4 Install Python Dependencies

Once the virtual environment is activated and you are in the project root directory (where `requirements.txt` is located), install all required packages with:

```bash
pip install -r requirements.txt 
```

**(This command installs the exact versions of TensorFlow, Keras, Streamlit, OpenCV, NumPy, Matplotlib, Seaborn, scikit-learn and other libraries needed to run the project.)**


To verify the installation, you can list the installed packages with:

```bash
pip list
```

---

## 3. Run stage1_data_prep.py and stage2_data_prep.py

 After the installations, run stage1_data_prep.py and stage2_data_prep.py py files using **"python stage1_data_prep.py"** and **"python stage2_data_prep.py"**. This splits the data into the necessary sub folders.
 The original dataset is now organized into two high-level classes:

- `no/` – non–brain-tumor MRIs  
- `yes/` – MRIs with a tumor present, further split into:
  - `glioma_tumor/`
  - `meningioma_tumor/`
  - `normal/`
  - `pituitary_tumor/`

---

## 4. Model Training

### 4.1 After running stage1_data_prep.py and stage2_data_prep.py, we can proceed to model training part. You can run the stage1_binary_training.py or download the '.h5' model file from here (https://www.dropbox.com/scl/fi/gz4ligs9gymu0hqxxhzmc/stage1_best_model.h5?rlkey=240l4jhnmkvyfg4qet4y3zdbm&st=ymiit21m&dl=0) and then you can proceed to download '.h5' model for stage 2 from here (https://www.dropbox.com/scl/fi/gz4ligs9gymu0hqxxhzmc/stage1_best_model.h5?rlkey=240l4jhnmkvyfg4qet4y3zdbm&st=d5n7ijk2&dl=0) or run the stage2_data_training.py 

### 4.2 Move the '.h5' models into the current working directory

**Note: In the dropbox link, it might show "unsupported file type", but it can still be downloaded.**

---

## 5. Model Testing

After training, you can proceed to run all the testing python scripts(stage1_binary_testing.py and stage2_testing.py) and then you can run the 'app_ui.py' script, which is the streamlit app using 'streamlit run app_ui.py'. This is the final app and the result of the work so far. (**The web page is slow but does work and load up**)

'streamlit run app_ui.py' will open the displayed local URL in your browser (typically http://localhost:8501).

Workflow in the app:

- Upload an MRI image (.jpg, .jpeg, or .png).
- Click "Run Diagnosis".
- View:
- Tumor / no tumor decision
- Tumor type (if applicable)
- Confidence scores
- Grad-CAM visualization highlighting discriminative regions (tumor cases only)

---
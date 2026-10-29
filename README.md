# Phishify360: A Comprehensive Multimodal Phishing Detection System

Phishify360 is a web-based phishing detection project that analyzes
three different website representations: **URLs, HTML content, and
webpage screenshots**. Each modality is processed by its own trained
model, and the web interface displays the prediction and confidence for
each available input.

> **Important:** The current implementation runs three independent
> classifiers. It does not combine their outputs into a single fused
> multimodal prediction. Predictions are model estimates, not a
> guarantee that a website is safe.

## Features

-   **URL analysis** using character-level TF-IDF features and Logistic
    Regression.
-   **HTML analysis** using character n-gram TF-IDF features and
    Logistic Regression.
-   **Screenshot analysis** using an EfficientNet-B0 image classifier.
-   **Flask web interface** for trying the models locally.
-   Dataset inspection, label verification, language auditing, training,
    and URL-model testing scripts in `src/`.

## Project Structure

``` text
Phishify360/
├── app.py
├── models/
│   ├── url_model.joblib
│   ├── html_model.joblib
│   └── efficientnet_b0_phish360.pth
├── src/
│   ├── check_labels.py
│   ├── check_languages.py
│   ├── check_unknown_labels.py
│   ├── inspect_dataset.py
│   ├── inspect_labels.py
│   ├── test_url_model.py
│   ├── train_html_model.py
│   ├── train_image_model.py
│   ├── train_url_model.py
│   └── verify_phish_labels.py
├── templates/
│   └── index.html
├── static/
├── dataset/             # Local dataset; not included in Git
├── .gitignore
└── README.md
```

## Technology Stack

-   Python
-   Flask
-   scikit-learn
-   TF-IDF text/vector features
-   Logistic Regression
-   PyTorch and Torchvision
-   EfficientNet-B0
-   HTML/CSS/JavaScript

## Getting Started

### 1. Clone the repository

``` bash
git clone https://github.com/Rudra310806/Phishyfy360.git
cd Phishyfy360
```

### 2. Create and activate a virtual environment

**Windows PowerShell:**

``` powershell
python -m venv .venv
.\.venv\Scripts\Activate.ps1
```

If PowerShell blocks activation, you can use the environment's Python
executable directly or follow your system's virtual-environment
activation guidance.

### 3. Install dependencies

Install the packages used by the application and models:

``` bash
python -m pip install --upgrade pip
pip install flask scikit-learn joblib pandas torch torchvision pillow
```

If your project has a tested `requirements.txt`, prefer installing from
it instead:

``` bash
pip install -r requirements.txt
```

### 4. Check the model files

Make sure the following trained model files exist in the `models/`
directory:

-   `url_model.joblib`
-   `html_model.joblib`
-   `efficientnet_b0_phish360.pth`

The dataset is not committed to the repository. Training scripts may
require the dataset and the paths expected by those scripts.

### 5. Run the web application

``` bash
python app.py
```

Open the local address shown by Flask in your terminal. In the current
local setup, this is expected to be:

``` text
http://127.0.0.1:5000
```

Use the interface to submit the supported input for each detector. Keep
the terminal open while the application is running.

## Model Overview and Reported Results

The following are the project's recorded development/validation results.
They are **not directly comparable**, because the models were evaluated
on different splits and with different procedures.

  ------------------------------------------------------------------------------
  Modality         Model                   Recorded result Evaluation context
  ---------------- ----------------- --------------------- ---------------------
  URL              Character               93.61% accuracy Internal validation
                   TF-IDF + Logistic                       split from the
                   Regression                              training/validation
                                                           data

  HTML             Character 3--5          96.22% accuracy Validation split
                   gram TF-IDF +                           
                   Logistic                                
                   Regression                              

  Screenshot       EfficientNet-B0         81.67% accuracy Preliminary image
                                                           holdout after two
                                                           training epochs
  ------------------------------------------------------------------------------

These results should be treated as preliminary. They do not establish
performance on every live website or on a fully independent, unseen test
set. Confidence values shown by a classifier should not automatically be
interpreted as calibrated probabilities.

## Dataset

The project used the **Phish360** dataset, which provides phishing and
legitimate website samples with URL, HTML, and screenshot information.

-   Official dataset page:
    <https://web.cs.hacettepe.edu.tr/~selman/phish360-dataset/>
-   The local dataset directory is excluded from Git to avoid committing
    large data files.
-   Follow the dataset provider's terms and citation guidance when using
    or redistributing the data.

A sample-based language audit was also run during development. It found
multiple language codes in the inspected sample, but this audit does
**not** prove that the trained models have equal detection accuracy
across languages.

## Development Scripts

Scripts in `src/` support dataset inspection, label checks, language
checks, model training, and testing. Run a script from the project root
so its relative paths resolve as expected, and review the script before
running it if your local dataset layout differs.

Examples:

``` bash
python src/inspect_dataset.py
python src/check_languages.py
python src/test_url_model.py
```

The exact dataset paths and required inputs depend on each script's
configuration.

## Limitations and Responsible Use

-   Phishing websites change frequently; model performance may degrade
    on new campaigns.
-   A legitimate prediction does not guarantee that a page is safe, and
    a phishing prediction may be a false positive.
-   The three classifiers currently operate independently; their results
    are not fused into one final decision.
-   Evaluation results are preliminary and use different data splits.
-   The language audit is sample-based and is not a multilingual
    accuracy benchmark.
-   Do not enter passwords, payment details, or other sensitive
    information into suspicious websites to test them.
-   This project is intended for educational and research purposes. Do
    not use it as the sole basis for security decisions.

## Repository

GitHub: <https://github.com/Rudra310806/Phishyfy360>

## Acknowledgements

Thanks to the Phish360 dataset authors and maintainers for making the
dataset available for research.

# Multimodal Retrieval-Augmented Generation (RAG)

This repository contains an implementation of a Multimodal RAG model capable of retrieving and generating responses based on both textual and visual inputs. The model processes text and image data to provide contextually rich answers. This README will guide you through the structure, installation, and usage of the project.

## Features
- **Multimodal Input**: Supports both text and image input.
- **Contextual Retrieval**: Fetches relevant information from a knowledge base.
- **Generative Responses**: Produces coherent and contextually appropriate answers.
- **PDF Output**: Generates a PDF file containing the input text, associated images, and model responses.

## Screenshots
1. **Input Text and Image**:
   ![Screenshot1](path/to/screenshot1.png)

2. **Knowledge Retrieval**:
   ![Screenshot2](path/to/screenshot2.png)

3. **Generated PDF**:
   ![Screenshot3](path/to/screenshot3.png)

## Installation

### Prerequisites
- Python 3.8 or higher
- Pip package manager
- Virtual environment (optional but recommended)

### Steps
1. Clone the repository:
   ```bash
   git clone https://github.com/your-username/multimodal-rag.git
   cd multimodal-rag
   ```

2. Create and activate a virtual environment:
   ```bash
   python -m venv env
   source env/bin/activate   # On Windows: env\Scripts\activate
   ```

3. Install dependencies:
   ```bash
   pip install -r requirements.txt
   ```

## Usage
1. Add your knowledge base and images to the appropriate folders.
2. Run the main script:
   ```bash
   python main.py --input_text "Your text here" --input_image path/to/image.jpg
   ```
3. The output will include a PDF file combining the input and response.

## Project Structure
```
multimodal-rag/
|-- data/
|   |-- knowledge_base/
|   |-- images/
|-- outputs/
|   |-- pdfs/
|-- src/
|   |-- models/
|   |-- utils/
|-- requirements.txt
|-- README.md
```

## Requirements
The required Python packages are listed below. Ensure they are installed in your environment:
- PyPDF2
- Pillow
- Transformers
- Sentence-Transformers
- Torch
- Scikit-learn
- PyMuPDF
- Pyngrok
- Tqdm

Alternatively, use the provided `requirements.txt` file to install all dependencies:
```text
PyPDF2
Pillow
Transformers
Sentence-Transformers
Torch
Scikit-learn
PyMuPDF
Pyngrok
Tqdm
```

## Contributions
Contributions are welcome! Feel free to fork the repository and submit a pull request.

## License
This project is licensed under the MIT License. See the LICENSE file for details.

---

For more details, please refer to the code and additional documentation in the repository. If you encounter any issues, create a new issue in the GitHub repository or contact the maintainers.


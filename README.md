# 🤖 MultiChat: Advanced PDF & Image Chat System

[![Python 3.8+](https://img.shields.io/badge/python-3.8+-blue.svg)](https://www.python.org/downloads/)
[![Contributions Welcome](https://img.shields.io/badge/contributions-welcome-brightgreen.svg?style=flat)](https://github.com/techyogeshchauhan/multichat/issues)

An innovative chat system that combines the power of PDF document interaction and image understanding. Chat naturally with your documents while leveraging visual context for richer, more meaningful conversations.

## ✨ Features

- 📑 **PDF Chat**
  - Upload and chat with multiple PDF documents
  - Smart context understanding and retrieval
  - Document summarization
  - Citation and source tracking

- 🖼️ **Image Processing**
  - Visual question answering
  - Image-based context enhancement
  - Multi-image support
  - OCR integration for text in images

- 🤝 **Multimodal Integration**
  - Seamless switching between document and image context
  - Combined understanding of text and visual elements
  - Context-aware responses
  - Rich, multimodal outputs

## 🚀 Getting Started

### Prerequisites

- Python 3.8 or higher
- Basic understanding of virtual environments
- GPU recommended for better performance

### Installation

1. **Clone the Repository**
```bash
git clone https://github.com/techyogeshchauhan/multichat.git
cd multichat
```

2. **Set Up Environment**
```bash
python -m venv venv
source venv/bin/activate  # On Windows: venv\Scripts\activate
```

3. **Install Dependencies**
```bash
pip install -r requirements.txt
```

## 💻 Usage

### Starting the Chat Interface

```bash
python app.py
```

Visit `http://localhost:8000` in your browser to access the chat interface.

### Basic Commands

```python
# Upload a PDF
/upload pdf_file.pdf

# Process an image
/image your_image.jpg

# Get document summary
/summary

# Clear context
/clear
```

## 🛠️ Architecture

```
multichat/
├── app/
│   ├── core/
│   │   ├── pdf_processor.py
│   │   ├── image_processor.py
│   │   └── chat_engine.py
│   ├── models/
│   │   └── model_config.py
│   └── utils/
│       └── helpers.py
├── static/
│   └── styles/
├── templates/
│   └── chat.html
└── app.py
```

## 📦 Key Dependencies

| Package | Purpose |
|---------|---------|
| PyPDF2 | PDF processing |
| Transformers | Language models |
| Pillow | Image processing |
| FastAPI | Web framework |
| Langchain | LLM integration |
| OpenAI | Language model API |

## 🎯 Use Cases

- **Academic Research**: Navigate through research papers and related images
- **Document Analysis**: Extract and discuss information from complex documents
- **Content Creation**: Generate content based on document and image inputs
- **Educational Support**: Interactive learning with visual and textual materials

## 🛣️ Roadmap

- [ ] Add support for more document formats
- [ ] Implement real-time collaboration
- [ ] Enhance visualization capabilities
- [ ] Add plugin system for extensibility
- [ ] Improve memory management for large documents

## 🤝 Contributing

Contributions are always welcome! Here's how you can help:

1. Fork the repository
2. Create your feature branch (`git checkout -b feature/AmazingFeature`)
3. Commit your changes (`git commit -m 'Add some AmazingFeature'`)
4. Push to the branch (`git push origin feature/AmazingFeature`)
5. Open a Pull Request

## 📝 License

Distributed under the MIT License. See `LICENSE` for more information.

## 📫 Contact

Yogesh Chauhan
- GitHub: [@techyogeshchauhan](https://github.com/techyogeshchauhan)
- LinkedIn: [https://www.linkedin.com/in/yogesh-chauhan-40650523b/]
- Email: [yogesh.chauhan.ai@gmail.com]

Project Link: [https://github.com/techyogeshchauhan/multichat](https://github.com/techyogeshchauhan/multichat)

---

<div align="center">

Made with ❤️ by [Yogesh Chauhan](https://github.com/techyogeshchauhan)

If this project helps you, please consider giving it a ⭐️!

</div>

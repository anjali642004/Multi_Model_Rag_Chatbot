# Multi-Media Chat Bot

## 🎯 What is My Project?

We built a **Multi-Media Chat Bot** - an intelligent assistant that can understand and respond to different types of content including text, PDFs, images, and audio files. Unlike traditional chatbots that only handle text, our bot can:

- **Read and analyze PDF documents** - Ask questions about uploaded PDFs
- **See and understand images** - Describe what's in uploaded pictures  
- **Listen to audio files** - Transcribe and answer questions about audio content
- **Chat naturally** - Have conversations while remembering context
- **Speak back** - Convert responses to audio for better user experience

## 🏗️ How We Built It

### Our Technical Approach

We created a **multimodal RAG (Retrieval-Augmented Generation)** system that combines:
- **Local AI models** (no cloud dependency)
- **Vector databases** for storing and retrieving information
- **Multiple processing pipelines** for different content types
- **Smart context management** for meaningful conversations

### Our Implementation Strategy

#### 1. **Frontend Interface (Streamlit)**
- Built a clean, user-friendly web interface
- File upload system for PDFs, images, and audio
- Real-time chat interface with message history
- Toggle switches for different modes (PDF chat, etc.)

#### 2. **AI Brain (Ollama + Llama3.2)**
- Used **Ollama** to run **Llama3.2:3b** model locally
- No internet dependency for AI processing
- Configured with optimal temperature (0.75) for balanced responses
- GPU acceleration for faster processing

#### 3. **Memory System (Chroma Vector Database)**
- **Chroma** for storing document embeddings locally
- Automatic fallback from Pinecone to Chroma (no API keys needed)
- Smart document chunking (512 characters with 64 overlap)
- Context retrieval for relevant information

#### 4. **Audio Processing Pipeline**
- **FFmpeg** for audio format conversion (to 16kHz mono WAV)
- **Google Speech Recognition** for audio-to-text
- **gTTS** for text-to-speech responses
- Automatic cleanup of temporary files

#### 5. **Document Processing**
- **PyPDF2** for PDF text extraction
- **RecursiveCharacterTextSplitter** for smart document chunking
- **OllamaEmbeddings** for creating document vectors
- **SQLRecordManager** for tracking processed documents

#### 6. **Image Understanding**
- **Visual Question Answering (VQA)** system
- Image analysis and description capabilities
- Integration with the main chat flow

## 🔧 Our Development Process

### Step 1: **Core Architecture Setup**
- Set up **Streamlit** as the web interface
- Integrated **LangChain** for LLM orchestration
- Configured **Ollama** with Llama3.2:3b model
- Set up **Chroma** vector database for local storage

### Step 2: **Multimodal Processing Implementation**
- **PDF Processing**: Built pipeline to extract, chunk, and embed PDF content
- **Audio Processing**: Integrated FFmpeg for audio conversion and speech recognition
- **Image Processing**: Added VQA capabilities for image understanding
- **Text Processing**: Implemented conversation memory and context management

### Step 3: **Smart Integration**
- Created **dual-chain system**: Regular chat vs RAG-enabled chat
- Implemented **context-aware retrieval** for relevant document chunks
- Added **conversation history** for follow-up questions
- Built **file management system** with automatic cleanup

### Step 4: **User Experience Optimization**
- **File upload handling** with proper validation
- **Real-time processing** with progress indicators
- **Audio responses** for better accessibility
- **Error handling** and user feedback systems

## 🎯 Key Features We Implemented

### **Smart Question Handling**
- **Context-Aware Responses**: The bot understands whether you're asking about uploaded content or general questions
- **Follow-up Questions**: Maintains conversation context for natural dialogue
- **Multi-Modal Integration**: Seamlessly switches between text, PDF, image, and audio processing

### **Advanced Processing Capabilities**
- **PDF Intelligence**: Extracts, chunks, and embeds PDF content for intelligent retrieval
- **Audio Understanding**: Converts audio to text and answers questions about audio content
- **Visual Analysis**: Processes images and answers visual questions
- **Conversation Memory**: Remembers previous interactions for coherent dialogue

### **User Experience Features**
- **Real-time Processing**: Shows progress indicators during file processing
- **Audio Responses**: Converts text responses to speech for accessibility
- **File Management**: Automatic cleanup of temporary files
- **Error Handling**: Graceful error messages and recovery

## 🚀 How to Run Our Project

### Quick Start
```bash
# Install dependencies
pip install -r requirements.txt

# Install FFmpeg (for audio processing)
# Windows: choco install ffmpeg
# Linux: sudo apt install ffmpeg
# macOS: brew install ffmpeg

# Install and start Ollama
ollama pull llama3.2:3b
ollama serve

# Run the application
streamlit run app.py
```

### Configuration
Our project uses `config.yaml` for model settings:
```yaml
chat_model:
  'model': "llama3.2:3b"
  'temperature': 0.75
  'num_gpu': 1
```

## 📁 Our Project Structure

```
Multimodal-RAG/
├── app.py                 # Main Streamlit application
├── config.yaml           # Model configuration
├── requirements.txt      # Python dependencies
├── src/
│   ├── ollama_chain.py   # LLM chains (chat & RAG)
│   ├── vectorstore.py    # Vector database management
│   ├── audio_processor.py # Audio processing pipeline
│   ├── pdf_handler.py    # PDF processing
│   ├── vqa.py           # Image processing
│   └── utils.py         # Utility functions
└── .cache/              # Temporary files and database
```

## 🔄 How Our Processing Works

### **PDF Processing Pipeline**
1. **Upload** → User uploads PDF files
2. **Extract** → PyPDF2 extracts text content
3. **Chunk** → Split into 512-character chunks with 64-character overlap
4. **Embed** → Create vector embeddings using Ollama
5. **Store** → Save to Chroma vector database
6. **Retrieve** → Find relevant chunks when user asks questions
7. **Answer** → Generate responses using retrieved context

### **Audio Processing Pipeline**
1. **Upload** → User uploads audio file (WAV/MP3)
2. **Convert** → FFmpeg converts to 16kHz mono WAV
3. **Transcribe** → Google Speech Recognition converts to text
4. **Combine** → Merge transcribed text with user's question
5. **Process** → LLM generates response based on audio content
6. **Respond** → Return answer and convert to speech

### **Smart Chain Selection**
- **Regular Chat**: Uses `OllamaChain` for general conversation
- **PDF Chat**: Uses `OllamaRAGChain` with document retrieval
- **Context Awareness**: Automatically selects appropriate processing method

## 🎯 What Makes Our Project Special

### **Local-First Approach**
- **No Cloud Dependency**: Everything runs on your local machine
- **Privacy-Focused**: Your documents and conversations stay private
- **Offline Capable**: Works without internet connection (except for speech recognition)

### **Intelligent Processing**
- **Context-Aware**: Understands the difference between asking about uploaded content vs general questions
- **Multi-Modal**: Seamlessly handles text, PDFs, images, and audio in one interface
- **Conversation Memory**: Maintains context for natural follow-up questions

### **Technical Innovation**
- **Dual-Chain Architecture**: Smart switching between regular chat and RAG-enabled chat
- **Vector Database Integration**: Efficient document storage and retrieval
- **Audio Pipeline**: Complete audio processing from upload to speech response
- **Error Handling**: Robust error management and user feedback

## 🏆 Our Achievements

### **Successfully Implemented**
✅ **Multimodal RAG System** - First-of-its-kind local implementation
✅ **Audio Processing Pipeline** - Complete audio-to-text-to-speech workflow  
✅ **PDF Intelligence** - Smart document analysis and Q&A
✅ **Image Understanding** - Visual question answering capabilities
✅ **Conversation Memory** - Context-aware dialogue system
✅ **User-Friendly Interface** - Clean, intuitive Streamlit web app

### **Technical Challenges Solved**
- **File Format Handling**: Robust processing of multiple file types
- **Memory Management**: Efficient vector storage and retrieval
- **Context Preservation**: Maintaining conversation flow across different content types
- **Error Recovery**: Graceful handling of processing failures
- **Performance Optimization**: Local processing with GPU acceleration

---

**Our Multi-Media Chat Bot** - A complete multimodal AI assistant built from scratch with local processing, privacy-first design, and intelligent content understanding!
#   M u l t i _ M o d e l _ R a g _ C h a t b o t  
 
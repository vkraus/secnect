# 🔐 Failed Login Detector

A Streamlit application that uses semantic similarity to identify potential failed login events in log files using Sentence-BERT embeddings.

![Demo](assets/demo.gif)

## 🌟 Features

- **Semantic Analysis**: Uses Sentence-BERT (all-MiniLM-L6-v2) model for semantic similarity
- **Text Normalization**: Automatically removes timestamps, IP addresses, and numbers
- **Interactive UI**: User-friendly interface with real-time analysis
- **Configurable Thresholds**: Adjust confidence levels for detection
- **Visual Analytics**: Distribution plots and threshold analysis
- **Export Options**: Download full results or high-confidence matches only

## 🚀 Getting Started

### Prerequisites

- Python 3.8 or higher
- pip

### Installation

1. Clone the repository:
```bash
git clone https://github.com/yourusername/failed-login-detector.git
cd failed-login-detector

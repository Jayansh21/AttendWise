# 👨‍💻 AttendWise - Smart Attendance Management System

<div align="center">


[![Python](https://img.shields.io/badge/Python-3.8+-3776AB?style=for-the-badge&logo=python&logoColor=white)](https://python.org)
[![Flask](https://img.shields.io/badge/Flask-2.0+-000000?style=for-the-badge&logo=flask&logoColor=white)](https://flask.palletsprojects.com/)
[![OpenCV](https://img.shields.io/badge/OpenCV-4.0+-5C3EE8?style=for-the-badge&logo=opencv&logoColor=white)](https://opencv.org/)
[![scikit-learn](https://img.shields.io/badge/scikit--learn-1.0+-F7931E?style=for-the-badge&logo=scikit-learn&logoColor=white)](https://scikit-learn.org/)

**🚀 Revolutionize attendance tracking with AI-powered face recognition technology**

[🌟 Overview](#-overview) •
[📋 Features](#-features) •
[🛠️ Installation](#️-installation) •
[🎯 Usage](#-usage) •
[📁 Project Structure](#-project-structure) •
[🏗️ Tech Stack](#️-tech-stack) •
[🤝 Contributing](#-contributing) •
[📄 License](#-license) •
[🙏 Acknowledgments](#-acknowledgments) •
[👨‍💻 Author](#-author)

</div>

---

## 🌟 Overview

**AttendWise** is a cutting-edge web-based attendance management system that harnesses the power of facial recognition technology to automate attendance marking in educational institutions and corporate environments. Built with modern technologies and a user-centric approach, AttendWise eliminates the hassle of manual attendance while ensuring accuracy and efficiency.

### 🎯 What makes AttendWise special?
- **⚡ Real-time Recognition**: Instant face detection and recognition using advanced ML algorithms
- **🎨 Modern Interface**: Clean, responsive web design that works on all devices
- **📊 Smart Analytics**: Automatic attendance logging with CSV export functionality
- **🔒 Secure & Reliable**: Robust face recognition using trained KNN models
- **🌐 Web-based**: No complex installations - runs directly in your browser

---

## ✨ Features

### 🎪 Core Functionality
- 📷 **Real-time Face Recognition**: Mark attendance instantly via webcam
- 👥 **User Management**: Easy registration with multiple face image capture
- 📋 **Attendance Tracking**: View and manage all registered users
- 🗑️ **Database Management**: Clear user database with delete functionality
- 📈 **Daily Reports**: Automatic CSV export of attendance records
- 📱 **Responsive Design**: Works seamlessly on desktop and mobile devices

### 🔧 Technical Features
- 🧠 **KNN Machine Learning**: Trained model for accurate face recognition
- 🎯 **OpenCV Integration**: Advanced computer vision capabilities
- ⚡ **Flask Backend**: Lightweight and scalable web framework
- 📊 **Data Export**: Structured attendance data in CSV format
- 🎨 **Modern UI/UX**: Intuitive and user-friendly interface

---

## 🛠️ Installation

### 📋 Prerequisites
Make sure you have the following installed on your system:

- 🐍 **Python 3.8+**
- 📦 **pip** (Python package manager)
- 📷 **Webcam** (for face recognition functionality)

### ⚙️ Setup Instructions

1. **📂 Clone the Repository**
   ```bash
   git clone https://github.com/Jayansh21/AttendWise.git
   cd AttendWise
   ```

2. **🔧 Create Virtual Environment** (Recommended)
   ```bash
   # Windows
   python -m venv venv
   venv\Scripts\activate

   # macOS/Linux
   python3 -m venv venv
   source venv/bin/activate
   ```

3. **📥 Install Dependencies**
   ```bash
   pip install -r requirements.txt
   ```

4. **🚀 Run the Application**
   ```bash
   flask run
   ```

5. **🌐 Access the Application**
   Open your browser and navigate to: `http://127.0.0.1:5000`

---

## 🎯 Usage

### 🚀 Getting Started

1. **👤 Register New Users**
   - Click on "Add New User"
   - Enter user details
   - Capture multiple face images for better recognition accuracy

2. **📸 Take Attendance**
   - Click "Take Attendance"
   - Allow webcam access
   - Position your face in front of the camera
   - Attendance will be marked automatically upon recognition

3. **👥 Manage Users**
   - Use "Show Users" to view all registered users
   - Use "Delete Users" to clear the database when needed

4. **📊 View Attendance Records**
   - Daily attendance records are automatically saved in the `Attendance/` folder
   - Files are saved in CSV format with date stamps

---

## 🏗️ Tech Stack

<div align="center">

| Category | Technologies |
|----------|-------------|
| **🔧 Backend** | ![Python](https://img.shields.io/badge/Python-3776AB?style=flat&logo=python&logoColor=white) ![Flask](https://img.shields.io/badge/Flask-000000?style=flat&logo=flask&logoColor=white) |
| **🤖 Machine Learning** | ![scikit-learn](https://img.shields.io/badge/scikit--learn-F7931E?style=flat&logo=scikit-learn&logoColor=white) ![OpenCV](https://img.shields.io/badge/OpenCV-5C3EE8?style=flat&logo=opencv&logoColor=white) |
| **📊 Data Processing** | ![pandas](https://img.shields.io/badge/pandas-150458?style=flat&logo=pandas&logoColor=white) ![NumPy](https://img.shields.io/badge/NumPy-013243?style=flat&logo=numpy&logoColor=white) |
| **🎨 Frontend** | ![HTML5](https://img.shields.io/badge/HTML5-E34F26?style=flat&logo=html5&logoColor=white) ![CSS3](https://img.shields.io/badge/CSS3-1572B6?style=flat&logo=css3&logoColor=white) ![JavaScript](https://img.shields.io/badge/JavaScript-F7DF1E?style=flat&logo=javascript&logoColor=black) |

</div>

---

## 📁 Project Structure

```
AttendWise/
├── 📁 static/                    # Static files (CSS, JS, images)
├── 📁 templates/                 # HTML templates
├── 📁 Attendance/                # Generated attendance CSV files
├── 📄 app.py                     # Main Flask application
├── 📄 requirements.txt           # Python dependencies
├── 📄 README.md                  # Project documentation
├── 📄 LICENSE                    # License file
└── 📄 .gitignore                 # Git ignore rules
```

---

## 🤝 Contributing

We welcome contributions from the community! Here's how you can help make AttendWise even better:

### 🎯 How to Contribute

1. **🍴 Fork the Project**
2. **🌿 Create your Feature Branch** (`git checkout -b feature/AmazingFeature`)
3. **💾 Commit your Changes** (`git commit -m 'Add some AmazingFeature'`)
4. **📤 Push to the Branch** (`git push origin feature/AmazingFeature`)
5. **🔄 Open a Pull Request**

### 💡 Ideas for Contributions
- 🎨 UI/UX improvements
- 🚀 Performance optimizations
- 📱 Mobile app development
- 🔒 Security enhancements
- 📚 Documentation improvements
- 🧪 Test coverage expansion

---

## 📄 License

This project is licensed under the MIT License - see the [LICENSE](LICENSE) file for details.

---

## 🙏 Acknowledgments

- **OpenCV** for powerful computer vision capabilities
- **Flask** for the lightweight yet robust web framework
- **scikit-learn** for machine learning implementation
- **The open-source community** for inspiration and resources

---

## 👨‍💻 Author

<div align="center">

**Jayansh Jain**

[![Email](https://img.shields.io/badge/Email-jjayansh1021%40gmail.com-red?style=for-the-badge&logo=gmail&logoColor=white)](mailto:jjayansh1021@gmail.com)
[![GitHub](https://img.shields.io/badge/GitHub-Jayansh21-black?style=for-the-badge&logo=github&logoColor=white)](https://github.com/Jayansh21)

</div>

---

<div align="center">

### 🌟 If you found this project helpful, please give it a star! ⭐

**Made with lots of ❤️**

</div>
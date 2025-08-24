# AttendWise - Face Recognition Based Attendance System

## Table of Contents
- [About The Project](#about-the-project)
- [Features](#features)
- [Getting Started](#getting-started)
  - [Prerequisites](#prerequisites)
  - [Installation](#installation)
- [Usage](#usage)
- [Contact](#contact)
- [Acknowledgments](#acknowledgments)

---

## About The Project

**AttendWise** is a web-based attendance management system utilizing face recognition technology to automate attendance marking in an efficient and user-friendly way. This system detects and recognizes faces using a trained KNN model and keeps accurate daily attendance logs.

Built with Python and Flask, OpenCV for face detection, and scikit-learn for machine learning, AttendWise aims to simplify the attendance process in schools, colleges, and corporate environments.

---

## Features

- Real-time face recognition attendance marking via webcam
- Add new users by capturing multiple face images
- View and manage registered users
- Clear user database with delete functionality
- Automatic daily export of attendance records (CSV)
- Responsive and intuitive web interface

---

## Getting Started

### Prerequisites

Make sure you have the following installed:

- Python 3.8+
- pip
- OpenCV (`opencv-python`)
- Flask
- scikit-learn
- pandas
- joblib
- numpy

---

### Installation

1. **Clone the repository:**
 ```bash
 git clone https://github.com/yourusername/attendwise.git
 cd attendwise
```
2. **(Optional) Create virtual environment and activate it:**
```bash
python -m venv venv
source venv/bin/activate # On Windows use venv\Scripts\activate
```

3. **Install dependencies:**
```bash
pip install -r requirements.txt
```

4. **Run the app:**
```bash
flask run
```
5. **Navigate to [http://127.0.0.1:5000](http://127.0.0.1:5000) to use AttendWise.**

---

## Usage

- Use **Add New User** to register individuals by capturing their face images.
- Click **Take Attendance** to start webcam face recognition and mark attendance automatically.
- Use **Show Users** to view the list of registered users.
- Use **Delete Users** to clear the registered users database.
- Attendance records will be saved daily in the `Attendance/` folder as CSV files.

---

## Contact

* Jayansh Jain - jjayansh1021@gmail.com
* Project Link: [https://github.com/Jayansh21/attendwise](https://github.com/yourusername/attendwise)

---

## Acknowledgments

- OpenCV for powerful computer vision capabilities  
- Flask for lightweight yet powerful web framework  
- scikit-learn for implementing the KNN face recognition model  
- Joblib for efficient model serialization  
- Inspiration and resources from various open-source projects  
- README template inspired by [othneildrew/Best-README-Template](https://github.com/othneildrew/Best-README-Template)



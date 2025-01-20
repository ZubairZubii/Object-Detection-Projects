
# 🚗 **Vehicle Detection, PPE Detection, and Poker Hand Card Detection** 🃏

## ⚡️ Overview ⚡️

Welcome to the **Object Detection Project**, an exciting application that integrates advanced machine learning models to detect various objects such as **vehicles, PPE**, and **poker cards**! 🚓👷‍♂️🃏

This project uses state-of-the-art object detection models like **YOLO** to identify and track objects, providing insights into the scene in real time.

### ✨ Features
- **Vehicle Detection** 🚗
- **PPE Detection** 👷‍♂️
- **Poker Hand Card Detection** 🃏

## 🧰 Skills & Technologies Used 🧰

![Python](https://img.shields.io/badge/Python-3.9-blue)
![OpenCV](https://img.shields.io/badge/OpenCV-4.5.4-green)
![YOLO](https://img.shields.io/badge/YOLO-v8.0-orange)
![Torch](https://img.shields.io/badge/Torch-1.7.0-red)
![CVZone](https://img.shields.io/badge/CVZone-1.5.6-yellow)

### 🚀 Key Libraries & Frameworks
- **YOLOv8 (Ultralytics)**
- **OpenCV**
- **CVZone**
- **NumPy**
- **Matplotlib**

## 🛠️ Requirements

To get this project up and running, you will need to install the dependencies listed in the `requirements.txt` file.

### Installation 📦

```bash
pip install -r requirements.txt
```

### Required Libraries:
```plaintext
cvzone==1.5.6
ultralytics==8.0.26
hydra-core>=1.2.0
matplotlib>=3.2.2
numpy>=1.18.5
opencv-python==4.5.4.60
Pillow>=7.1.2
PyYAML>=5.3.1
requests>=2.23.0
scipy>=1.4.1
torch>=1.7.0
torchvision>=0.8.1
tqdm>=4.64.0
filterpy==1.4.5
scikit-image==0.19.3
lap==0.4.0
```

## 📸 Screenshots 📸

> Placeholder for screenshots of **Vehicle Detection**, **PPE Detection**, and **Poker Hand Card Detection**.

- **Vehicle Detection:**
  
  ![Image](https://github.com/user-attachments/assets/ab189b82-e19b-44e3-a87a-4abb210dd8c4)

  ![Image](https://github.com/user-attachments/assets/9b3f9565-c0e3-4215-91f5-66649fd00827)

  ![Image](https://github.com/user-attachments/assets/84c7a785-47b6-4bcc-8fc9-077317c0ad9e)

  
- **PPE Detection:**
  
  ![Image](https://github.com/user-attachments/assets/d0c49c11-8688-43be-af76-4ec159ad33d5)


- **Poker Hand Card Detection:**
  
  ![Image](https://github.com/user-attachments/assets/41284f06-9a98-4d80-9bd4-18192b117635)

  ![Image](https://github.com/user-attachments/assets/ae0cd534-b9b1-479c-846a-d42b709b183b)

## 🎮 Demo

Check out the demo of the **Vehicle Detection** model:

```python
cap = cv2.VideoCapture("cars.mp4")  # For Video

model = YOLO("yolov8l.pt")

while True:
    success, img = cap.read()
    if not success:
        print("Failed to read video frame. Exiting...")
        break

    results = model(img, stream=True)
    # Process detections here
```

## 🔧 How It Works

### 🚗 **Vehicle Detection**

This model detects cars, trucks, buses, motorbikes, and more! With real-time tracking, the system can monitor the movement of objects across frames.

- **Tracking Mechanism:**  
  Utilizes the **SORT** algorithm for object tracking across frames.

- **Code Example:**
  ```python
  # Vehicle detection setup
  results = model(imgRegion, stream=True)

  for r in results:
      boxes = r.boxes
      # Loop through each detected object
  ```

### 👷‍♂️ **PPE Detection**

Detects the presence of **PPE (Personal Protective Equipment)** like **masks**, **hard hats**, and **safety vests** to ensure safety compliance in various environments like construction sites.

- **Real-time Detection:**  
  Using a YOLO model fine-tuned for PPE detection.

- **Code Example:**
  ```python
  cap = cv2.VideoCapture(0)  # For Webcam
  model = YOLO("best.pt")

  while True:
      success, img = cap.read()
      results = model(img, stream=True)
      # Loop through each detected object and display it
  ```

### 🃏 **Poker Hand Card Detection**

Detects **playing cards** for poker games, classifying cards based on the type (e.g., 2 of Hearts, Ace of Spades).

- **Model:**  
  A custom-trained YOLO model detects different cards in real-time.

- **Code Example:**
  ```python
  cap = cv2.VideoCapture(0)  # For Webcam
  model = YOLO("playingCards.pt")
  
  while True:
      success, img = cap.read()
      results = model(img, stream=True)
      # Process the card detections
  ```

---

## 👩‍💻 Contribution

We welcome contributions! Feel free to fork the repository, raise issues, or submit pull requests. Let's build something amazing together!

## 📧 Contact

For any queries or feedback, feel free to reach out to me:

- **Email:** zs970120@gmail.com
- **GitHub:** [ZubairZubii](https://github.com/ZubairZubii)

---

**Credits:**
- **[Ultralytics YOLOv8](https://github.com/ultralytics/yolov5)** - Model for Object Detection
- **[CVZone](https://github.com/cvzone/cvzone)** - Image Processing
- **[PokerHandFunction](https://github.com/yourusername/poker-hand-function)** - Custom Poker Hand Recognition

---

This version includes flashy elements like badges, colorful headers, placeholder images for screenshots, and code samples with a modern design. It also provides a professional structure to the README file. Let me know if you want any further customizations!

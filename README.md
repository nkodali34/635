# 635
**Group Project: Smart Doorbell using Raspberry Pi and ML**

**Motivation:** Adithya Kumar, Naga Sujay Kodali and Indra Murala, have decided to work on the project “Smart Doorbell using Raspberry Pi and Machine Learning.” For Adithya, the appeal of this project lies in its hands-on nature — working directly with sensors and circuits allows me to immediately see the results of my efforts and explore how even small design choices in hardware can influence the overall behavior of a system. Sujay brings a physical-design background and is excited to gain a new perspective on how hardware bottlenecks affect software performance, sharpen his optimization mindset, and tackle compute, memory, and power constraints when deploying ML models on a Raspberry Pi. Indra is motivated by his passion for applying machine learning to less powerful hardware and is eager to experiment with optimizing models to run efficiently in resource-constrained environments while analyzing trade-offs such as memory usage and accuracy. Together, we see this project as an opportunity to combine our strengths, develop a hardware-aware approach to ML deployment, and test our ideas directly on edge hardware to create something both practical and innovative.

**Design goals:** Use Raspberry Pi with a camera module to design a smart doorbell that can detect people and classify whether they are recognized (e.g., known household members) or unknown.

**Deliverables:**
• Learn how to deploy ML models on Raspberry Pi
• Implement a lightweight face detection or person detection model using TensorFlow Lite.
• Build a simple system that, when someone appears at the door, captures an image and classify.
• Optional: Add an alert mechanism (e.g., send a notification to a phone or log it in a file).
• Fun feature!: Can it detect a delivery person? E.g., pizza or package delivery worker.
• The final output should be a code snippet and demonstration running inference directly on Raspberry Pi.

**System blocks:** Rasberry Pi, Cameera Module, Google Colab. You can use any model for face detection. There are various approaches. One simple way is to compute embeddings of known/trusted people using a CNN and store them. When a person approaches the door, compare their embedding with known embeddings. If the embeddings match, you know it is a trusted person.

**HW/SW requirements:** Raspberry Pi, camera module. For training, Google Colab can be used.

**Team members responsibilities:** 
• Adithya Kumar: Networking, Writing
• Naga Sujay Kodali: Setup, Software
• Indra Murala: Research, Algorithm Design

**Project timeline:** 
Week 1: Paper reading and Background Work.
Week 2: Implementing the camera module to get data from the raspberry pi for model training.
Week 3-4: Designing and training the model for facial detection. Collect dataset & preprocess.
Week 5: Fine tuning the modal.
Week 6: Deploying the model on Raspberry pi and testing.
Week 7: Report.

**References:** 
A lightweight CNN paper on edge vision.
TensorFlow Lite documentation for Raspberry Pi deployment.



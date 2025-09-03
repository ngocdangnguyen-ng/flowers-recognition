# Flowers Recognition Project - Personal Deep Learning Project
## Project Overview
This is a personal learning project where I explore deep learning techniques for image classification, specifically flower recognition. As a computer science student passionate about artificial intelligence and data science, I built this project to practice the full deep learning workflow and share my experience with others. The project covers all key steps of a modern ML pipeline, from data exploration and preprocessing to model development, evaluation, and deployment.

## Objective:
To develop an accurate image classification model for flower species using real-world data, while gaining hands-on experience with the end-to-end deep learning workflow.

## Key Result
| Model                      | Accuracy | Loss  |        Notes                 |
|----------------------------|----------|-------|------------------------------|
| Improved CNN               | 38.9%    | 2.4473|Custom architecture, optimized|
| Transfer Learning ResNet50 | 55.6%    | 1.1129| Pretrained, fine-tuned       |
| ResNet50 Fine-tuned        | 61.6%    | 1.0330|Further optimized, best result|

The best-performing model, ResNet50 Fine-tuned, reached ~62% accuracy. This shows the benefit of transfer learning over a custom CNN, but also highlights current limitations: the dataset is small and imbalanced, flower species share many visual similarities, and only ResNet50 was tested. While promising, the results leave room for improvement with stronger architectures (e.g., EfficientNet, ViT), better augmentation, and more advanced optimization.

## Features
* End-to-end workflow: EDA, preprocessing, data splitting, training, evaluation, prediction
* Modular, reusable codebase
* Multiple architectures: Custom CNN, Transfer Learning (ResNet50)
* Automated resizing, normalization, and data augmentation
* Professional visualizations and reporting
* Models saved as .h5 and pipelines for fast prediction
* Step-by-step notebooks for easy learning and extension

## Getting Started
**Installation**
```
git clone https://github.com/ngocdangnguyen-ng/flowers-recognition.git
cd flowers-recognition
pip install -r requirements.txt
```
**Quick Example**
```
from tensorflow.keras.models import load_model
from tensorflow.keras.preprocessing import image
import numpy as np

model = load_model('models/transfer_resnet50_finetuned_best.h5')
img = image.load_img('data/sample_images/daisy.jpg', target_size=(224,224))
x = image.img_to_array(img) / 255.0
x = np.expand_dims(x, axis=0)
pred = model.predict(x)
class_names = ['daisy', 'dandelion', 'rose', 'sunflower', 'tulip']
print(f"Predicted class: {class_names[np.argmax(pred)]}")
```

## Project Structure
```
flowers-recognition/
│
├── app/                     
├── backups/                 
├── configs/                 
├── data/                    
│   ├── raw/                 
│   └── processed/           
├── models/                  
│   ├── Improved_CNN_best.h5
│   ├── Transfer_ResNet50_best.h5
│   └── Transfer_ResNet50_FineTuned_best.h5
├── notebooks/               
│   ├── 01_exploratory_data_analysis.ipynb
│   ├── 02_data_preprocessing.ipynb
│   ├── 03_model_training.ipynb
│   ├── 04_model_evaluation.ipynb
│   └── 05_prediction_demo.ipynb
├── results/                
├── src/                    
│   ├── __init__.py
│   ├── data/
│   ├── evaluation/
│   ├── models/
│   └── utils/
├── tests/                  
├── evaluation/              
├── LICENSE                  
├── requirements.txt         
├── Dockerfile               
├── docker-compose.yml      
├── .gitignore               
├── README.md               
```

## Process Overview
1. **EDA:** Analyze data, check image quality, class distribution
2. **Preprocessing:** Resize, normalize, augment, split into train/val/test
3. **Modeling:** Train custom CNN and Transfer Learning with ResNet50
4. **Evaluation:** Assess accuracy, visualize confusion matrix, ROC
5. **Prediction:** Demo predictions on new images

## What I Learned & Challenges
**Technical Skills:**
- Data Pipeline: Built robust preprocessing pipeline, handled corrupted images
- Transfer Learning: Experienced firsthand how pretrained models outperform custom architectures
- Model Optimization: Learned to balance model complexity and generalization
- Evaluation: Implemented comprehensive model comparison and error analysis

**Key Insights:**
- Transfer learning provided a 21.8% accuracy boost over custom CNN
- Data augmentation was crucial for improving generalization on small dataset
- Fine-tuning pretrained models requires careful learning rate adjustment
- Visual similarity between flower species creates natural classification limits

## Limitations & Future Work
**Current Limitations:**
- Limited to 5 common flower species
- Small dataset affects model robustness
- Only tested ResNet50 architecture
- Some flower classes remain difficult to distinguish

**Planned Improvements:**
- Test modern architectures (EfficientNet, Vision Transformer)
- Implement k-fold cross-validation for better model selection
- Add model interpretability with Grad-CAM visualization
- Create web application for easy demonstration
- Experiment with advanced data augmentation techniques
- Collect additional training data

## Acknowledgments
* **Courses:** Andrew Ng's Deep Learning Specialization
* **Datasets:** Kaggle Flowers Recognition Dataset
* **Tools:** TensorFlow, Keras, scikit-learn, matplotlib, seaborn

## Contact
* **LinkedIn:** https://www.linkedin.com/in/ngocnguyen-fr
* **Email:** nndnguyen2016@gmail.com

---

I welcome feedback and suggestions for improvement. Thank you for visiting my project!

## License
This project is licensed under the MIT License. See the LICENSE file for details.
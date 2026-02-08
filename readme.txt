#  Intrusion Detection System with Machine Learning

Network intrusion detection system using machine learning to classify network traffic and detect cyber attacks.

##  About

This project implements a machine learning-based IDS trained on the **CICIDS2017 dataset** to detect and classify different types of network attacks in real-time.

##  Dataset

The project uses three datasets from CICIDS2017:
- **Monday-WorkingHours.pcap_ISCX.csv** - Normal network traffic
- **Friday-WorkingHours-Afternoon-DDos.pcap_ISCX.csv** - DDoS attacks
- **Friday-WorkingHours-Afternoon-PortScan.pcap_ISCX.csv** - Port scan attacks

Download the datasets from: [CICIDS2017](https://www.unb.ca/cic/datasets/ids-2017.html)

Place the CSV files in the `data/` folder.

##  Technologies

- **Python 3.8+**
- **Scikit-learn** - Machine learning models
- **Pandas** - Data processing
- **Matplotlib/Seaborn** - Visualization

##  Features

- **Binary Classification**: Normal vs Attack traffic
- **Multi-class Classification**: BENIGN, DDoS, PortScan
- **Feature Selection**: Automatic selection of most important network features
- **Model Comparison**: Multiple ML algorithms with performance evaluation

##  Machine Learning Models

- **Random Forest Classifier**
- **Decision Tree Classifier**

Both models are trained to classify network traffic as:
- BENIGN (Normal)
- DDoS
- PortScan

 📁 Project Structure
```
IDS-Machine-Learning/
├── data/                    # Datasets (CSV files)
├── src/                     # Source code
│   ├── data_explore01.py
│   ├── data_process02.py
│   ├── data_features03.py
│   ├── data_train_model04.py
│   └── data_model_evaluation05.py
├── results/                 # Visualizations and metrics
└── docs/                    # Documentation
```

##  Usage
0. Create the folders 
1. Download CICIDS2017 datasets
2. Place CSV files in `data/` folder
3. Run the scripts created in /src in order:
```bash
python src/data_explore01.py
python src/data_process02.py
python src/data_features03.py
python src/data_train_model04.py
python src/data_model_evaluation05.py
```

##  Results

The models achieve high accuracy in detecting network intrusions. See the `results/` folder for detailed performance metrics and visualizations.

##  Author

**Younes**


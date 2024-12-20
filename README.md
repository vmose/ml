# ML  

This repository contains a collection of Python scripts that are designed for a variety of analytical projects. Each script showcases the use of powerful data analysis and machine learning libraries to address different problems and visualize results.  

## Libraries Used  

The following Python libraries were utilized in these projects:  

- **pandas**: Data manipulation and analysis.  
- **numpy**: Numerical computing.  
- **matplotlib**: Visualization and plotting.  
- **seaborn**: Statistical data visualization.  
- **scikit-learn**: Machine learning models and utilities.  
- **torch**: Deep learning and neural networks.
- **requests**: Fetching data from a RESTful API or GraphQL API.
    
## Features  

- Each script is organized with `# %%` markers for executable sections. This allows users to execute the code line by line in any compatible code editor, such as:  
  - [VS Code](https://code.visualstudio.com/)  
  - [PyCharm](https://www.jetbrains.com/pycharm/)  
  - [Jupyter Notebook](https://jupyter.org/)  
- The projects include a variety of tasks, such as:  
  - Data cleaning and preprocessing.  
  - Exploratory data analysis (EDA).  
  - Machine learning model training and evaluation.  
  - Visualization of data and results.  

## How to Use  

1. Clone this repository:  
   ```bash  
   git clone https://github.com/vmose/ml.git  
   cd python_notebooks
2. Install the required libraries:
    ```bash
    pip install -r requirements.txt  
3. Open any .py file in your preferred code editor. Use the # %% markers to execute each section of the script.

## Projects Overview  

| Script Name         | Description                               | Key Libraries Used          |
|---------------------|-------------------------------------------|-----------------------------|
| `kiva_analysis.py`  | Deep Analysis of the kiva dataset imported via kiva.org API. | pandas, requests, matplotlib |
| `matplotlib.py`     | Data exploration & visualization with matplotlib.            | matplotlib, seabon, numpy    |
| `seaborn.py`        | Data exploration & visualization with seaborn.               | matplotlib, seabon, numpy    |
| `numpy.py`          | Standard data analyses with numpy                            |  numpy                       |
| `pandas.py`         | Standard data analyses with pandas                           | pandas, numpy                |
| `torch.py`          | Data exploration using torch for deep learning               | torch, numpy                 |
| `oop_functions.py`  | Standard data functions                                      | python3                      |
| `oop_with_python.py`  | Object Oriented Classes & Functions                        | python3                  	  |
| `polygon_transformations.py`  | Geoometrics transformations with shapely           | pandas, shapely, json        |
| `pytube.py`         | Downloading mp3/4 files using pytube                         | pytube                       |
| `sample_automation.py`  | Automated python file for ETL extraction & reporting     | pandas, sqlachemy            |
| `heart_disease.py`  | Using data science to explore the symptoms of heart disease  | pandas, sqlachemy            |
| `doctors_note.py`   | Build a fake doctor's note to get out of work (WHICH I WOULD NEVER DO)       | fpdf         |




   Feel free to explore, modify, and use the scripts to fit your needs! Contributions are welcome.

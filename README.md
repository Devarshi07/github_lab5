# F1 Race Data Analyzer - MLOps Lab 1

A Formula 1 race data analysis project with machine learning capabilities, replacing the traditional calculator implementation for MLOps Lab 1 assignment.

## 🏎️ Project Overview

This project demonstrates MLOps principles through real-world F1 data analysis:
- **Data Processing**: Load and analyze F1 race data using FastF1 API
- **Statistical Analysis**: Calculate lap times and compare driver performance
- **Machine Learning**: Linear regression model for lap time prediction
- **Testing**: Comprehensive test suites with pytest and unittest
- **CI/CD**: Automated testing via GitHub Actions

## 📊 Key Differences from Original Project

| Aspect | Original (Calculator) | This Project (F1 Analyzer) |
|--------|----------------------|----------------------------|
| **Functionality** | Basic arithmetic operations | F1 data analysis with ML |
| **Libraries** | None (pure Python) | FastF1, pandas, numpy, scikit-learn |
| **Complexity** | 4 simple functions | 5 complex analysis functions |
| **Machine Learning** | ❌ None | ✅ Linear Regression model |
| **Real-world Data** | ❌ Simple numbers | ✅ Live F1 race data |

## 🎯 Features

1. **`load_session_data()`** - Load F1 race/qualifying/practice session data
2. **`get_lap_times()`** - Extract lap times for specific drivers
3. **`calculate_average_lap_time()`** - Calculate average lap times
4. **`predict_lap_times()`** - ML model to predict future lap times
5. **`compare_drivers()`** - Compare performance between two drivers

## 📁 Project Structure
```
github/
├── .github/workflows/
│   ├── pytest_action.yml
│   └── unittest_action.yml
├── src/
│   ├── __init__.py
│   └── f1_analyzer.py
├── test/
│   ├── __init__.py
│   ├── test_pytest.py (9 tests)
│   └── test_unittest.py (10 tests)
├── requirements.txt
└── README.md
```

## 🚀 Quick Start

### 1. Setup Virtual Environment
```bash
python -m venv lab_01
source lab_01/bin/activate  # Mac/Linux
# or
lab_01\Scripts\activate     # Windows
```

### 2. Install Dependencies
```bash
pip install -r requirements.txt
```

### 3. Run Tests
```bash
# Pytest
pytest test/test_pytest.py -v

# Unittest
python -m unittest test.test_unittest
```

## 📦 Dependencies
```
fastf1>=3.1.0
pandas>=2.0.0
numpy>=1.24.0
scikit-learn>=1.3.0
pytest>=7.4.0
```

## 🧪 Testing

- **Pytest**: 9 tests with fixtures and parametrized tests
- **Unittest**: 10 tests with setUp method and comprehensive assertions
- **Coverage**: Edge cases, error handling, ML model validation
- **CI/CD**: Automated testing on every push/PR to main branch

## 🔄 GitHub Actions

Both workflows trigger on push/PR to main:
- **pytest_action.yml** - Runs pytest tests, generates XML report, uploads artifacts
- **unittest_action.yml** - Runs unittest tests with success/failure notifications

## 🎓 Key Learning Outcomes

✅ Virtual environment management  
✅ Git version control and .gitignore  
✅ Structured project organization  
✅ Pytest and unittest frameworks  
✅ GitHub Actions CI/CD pipeline  
✅ Machine learning integration  
✅ Real-world data processing  


---

**Author**: Devarshi Mahajan  
**Institution**: Northeastern University  

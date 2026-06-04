# Financial Time Series Forecasting using SAX and Association Rules

The approach focuses on discovering recurring patterns in price dynamics and transforming them into interpretable trading signals: BUY, SELL, or HOLD.

The model converts a time series into a symbolic representation and extracts association rules that capture underlying patterns in the data.

---

## Motivation

Forecasting financial time series remains a challenging task due to noise, non-stationarity, and changing market regimes.

While many modern approaches rely on statistical models or machine learning, this project investigates whether symbolic representations of market behavior can reveal recurring patterns useful for forecasting future price movements.

The proposed method combines SAX discretization and association rule mining to transform historical price dynamics into interpretable trading signals.

---

## Live Demo

Try the application online:

https://aappciation-rules-forecasting-24o6pfr8qmtezadbukxdp8.streamlit.app/

---

## Description

The method combines signal processing and symbolic representation techniques:

1. The original time series is filtered using the Hodrick–Prescott (HP) filter to extract the cyclical component

2. The signal is smoothed using a moving average

3. The series is normalized and discretized using SAX

4. Symbolic sequences are used to build association rules of the form:

   abcde → f

5. For each rule, statistical characteristics are calculated:

   * support
   * confidence
   * lift
   * expected return
   * probability of upward and downward movement

6. Trading signals are generated based on:

   * probability thresholds
   * expected return
   * rule strength

The method is evaluated using walk-forward validation and demonstrates consistent predictive performance across different market conditions.

---

## Methodology

The forecasting pipeline consists of the following stages:

### 1. Trend-Cycle Decomposition

The original financial time series is processed using the Hodrick–Prescott (HP) filter to isolate the cyclical component and reduce the influence of long-term trends.

### 2. Signal Smoothing

A moving average is applied to reduce short-term fluctuations and suppress market noise.

### 3. Symbolic Representation

The processed series is normalized and transformed into a Symbolic Aggregate approXimation (SAX) representation. Continuous numerical values are converted into discrete symbols, allowing the time series to be analyzed as a sequence of patterns.

### 4. Association Rule Mining

Sliding symbolic windows are used to generate association rules of the form:

```text
ABCDE → F
```

Each rule represents a recurring relationship between historical patterns and subsequent market behavior.

### 5. Rule Evaluation

For every extracted rule, the following statistical measures are calculated:

- support
- confidence
- lift
- expected return
- probability of upward movement
- probability of downward movement

### 6. Signal Generation

Trading recommendations (BUY, SELL, or HOLD) are generated using a combination of:

- rule confidence
- expected return
- probability thresholds
- rule strength metrics

### 7. Walk-Forward Validation

Model performance is evaluated using walk-forward validation, ensuring that only historical information is available during each prediction step and preventing data leakage.

### Forecasting Pipeline

```text
Financial Time Series
          ↓
      HP Filter
          ↓
   Moving Average
          ↓
    Normalization
          ↓
 SAX Representation
          ↓
Association Rules
          ↓
 Signal Generation
          ↓
Walk-Forward Validation
```

---

## Experimental Results

The proposed approach was evaluated on four publicly traded companies from different markets and sectors:

- Yandex (Technology, Russia)
- Apple (Technology, USA)
- GSK (Pharmaceuticals, United Kingdom)
- Cipla (Pharmaceuticals, India)

The experiments were conducted using walk-forward validation. Different combinations of rule length (`k`) and forecasting horizon (`H`) were evaluated to analyze the impact of model parameters on forecasting performance.

### Yandex

| k | H | Directional Accuracy | Overall Accuracy | Average Return (%) |
|---|---|---------------------|------------------|--------------------|
| 3 | 5 | 0.639 | 0.694 | 1.07 |
| 5 | 5 | 0.688 | 0.736 | 0.69 |
| 7 | 5 | 0.688 | 0.746 | 0.40 |
| 3 | 10 | 0.705 | 0.663 | 1.91 |
| 5 | 10 | 0.772 | 0.648 | 1.17 |
| 7 | 10 | 0.758 | 0.623 | 0.73 |

For Yandex, the highest directional accuracy (0.772) was achieved for k = 5 and H = 10, while the highest average return (1.91%) was obtained for k = 3 and H = 10.

### Apple

| k | H | Directional Accuracy | Overall Accuracy | HOLD Ratio | Average Return (%) |
|---|---|---------------------|------------------|------------|--------------------|
| 5 | 5 | 0.748 | 0.814 | 0.680 | 0.69 |
| 7 | 5 | 0.773 | 0.831 | 0.828 | 0.42 |
| 5 | 10 | 0.812 | 0.718 | 0.679 | 1.20 |
| 7 | 10 | 0.828 | 0.686 | 0.826 | 0.66 |
| 8 | 10 | 0.843 | 0.674 | 0.889 | 0.44 |

For Apple, increasing the rule length improved directional accuracy from 0.748 to 0.843, but also increased the proportion of HOLD signals and reduced strategy returns.

### GSK

| k | H | Directional Accuracy | Overall Accuracy | HOLD Ratio | Average Return (%) |
|---|---|---------------------|------------------|------------|--------------------|
| 5 | 5 | 0.696 | 0.882 | 0.739 | 0.42 |
| 7 | 5 | 0.757 | 0.921 | 0.874 | 0.26 |
| 5 | 10 | 0.753 | 0.816 | 0.742 | 0.65 |
| 7 | 10 | 0.813 | 0.824 | 0.873 | 0.39 |
| 8 | 10 | 0.833 | 0.824 | 0.907 | 0.30 |

For GSK, the highest directional accuracy (0.833) was obtained for k = 8 and H = 10, while the highest average return (0.65%) was achieved for k = 5 and H = 10.

### Cipla

| k | H | Directional Accuracy | Overall Accuracy | HOLD Ratio | Average Return (%) |
|---|---|---------------------|------------------|------------|--------------------|
| 5 | 5 | 0.429 | 0.214 | 0.951 | 0.06 |
| 7 | 5 | 0.333 | 0.206 | 0.979 | 0.07 |
| 5 | 10 | 0.559 | 0.344 | 0.951 | 0.36 |
| 7 | 10 | 0.611 | 0.346 | 0.974 | 0.21 |
| 8 | 10 | 0.600 | 0.344 | 0.986 | 0.11 |

For Cipla, the proposed method demonstrated substantially weaker predictive performance compared to the other evaluated assets. Although longer forecasting horizons improved directional accuracy, the model generated a very high proportion of HOLD signals, indicating limited confidence in actionable predictions for this stock.

### Summary of Best Directional Accuracy Configurations

| Company | k | H | Directional Accuracy | Overall Accuracy | Average Return (%) |
|----------|---|---|---------------------|------------------|--------------------|
| Yandex | 5 | 10 | 0.772 | 0.648 | 1.17 |
| Apple | 8 | 10 | 0.843 | 0.674 | 0.44 |
| GSK | 8 | 10 | 0.833 | 0.824 | 0.30 |
| Cipla | 7 | 10 | 0.611 | 0.346 | 0.21 |

### Discussion

The experiments demonstrate that the proposed SAX-based association rule approach can achieve strong directional forecasting performance for several assets, particularly Apple, GSK, and Yandex.

Across multiple datasets, increasing rule length generally improved directional accuracy but also increased the proportion of HOLD signals, resulting in a trade-off between prediction confidence and trading activity.

The highest directional accuracy was achieved for Apple (0.843), while the highest average strategy return was observed for Yandex (1.91%) at k = 3 and H = 10.

The results obtained for Cipla indicate that the effectiveness of symbolic pattern mining depends on asset-specific characteristics and may require additional parameter tuning for different markets and instruments.

---

## Example Association Rules

One of the advantages of the proposed approach is interpretability. Instead of producing opaque predictions, the model generates explicit symbolic patterns associated with future market movements.

Examples of high-scoring rules extracted for Apple (k = 5, H = 10):

| Rule | Confidence | Lift | Expected Return | Action |
|--------|------------|------|----------------|--------|
| ggggg → g | 0.863 | 7.908 | -4.66% | SELL |
| aaaaa → a | 0.855 | 7.858 | +4.43% | BUY |
| bbbbb → b | 0.740 | 9.655 | +3.27% | BUY |

These rules indicate recurring symbolic patterns in historical price behavior. High confidence and lift values suggest that the observed patterns occur significantly more often than expected by chance and can therefore be used as the basis for trading recommendations.

---

## Example Datasets

Sample datasets used in the experiments are available in the `data/` directory.

- Yandex.csv
- Apple.csv
- GSK.csv
- Cipla.csv

These datasets can be used to reproduce the results presented in the Experimental Results section.

---

## Streamlit Application

The repository includes an interactive web application that allows:

* uploading a CSV file with time series data
* configuring model parameters
* viewing predictions and evaluation metrics
* exploring generated association rules

---

### Main Screen
![Main](screenshots/interface.png)
The main interface of the application with parameter configuration and CSV file upload.

### Time Series Visualization
![Chart](screenshots/chart.png)
Visualization of the processed time series after filtering and smoothing.

### Prediction Metrics
![Metrics](screenshots/metrics.png)
Evaluation metrics including accuracy of predictions and strategy performance.

### Trading Signal
![Signal](screenshots/recommendation.png)
Final trading recommendation (BUY, SELL, or HOLD) generated by the model.

### Association Rules
![Rules](screenshots/rules.png)
Top association rules extracted from the symbolic time series representation.

## System Design

### Use Case Diagram
![Use Case Diagram](screenshots/uml.png)
Use case diagram illustrating the functionality of the web application.

---

## Project Structure

```text
.
├── data/
│   ├── Apple.csv
│   ├── Cipla.csv
│   ├── GSK.csv
│   └── Yandex.csv
├── screenshots/
│   ├── interface.png
│   ├── chart.png
│   ├── metrics.png
│   ├── recommendation.png
│   ├── rules.png
│   └── uml.png
├── app.py
├── main.py
├── elbow_method.py
├── requirements.txt
├── .gitignore
└── README.md
```

## Installation

Install dependencies:

```bash
pip install -r requirements.txt
```

---

## Run

To run the web application:

```bash
streamlit run app.py
```

---

## Requirements

* Python 3.9+
* pandas
* numpy
* scipy
* matplotlib
* streamlit

---

## Notes

* The application interface is in English
* The method can be applied to different types of time series
* The project demonstrates an experimental approach to time series forecasting based on symbolic representation and pattern mining

---

## Author

Ekaterina Polosmak


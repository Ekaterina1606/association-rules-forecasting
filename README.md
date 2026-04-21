# Financial Time Series Forecasting using SAX and Association Rules

This project implements a method for forecasting financial time series using SAX (Symbolic Aggregate approXimation) and association rules.

The model transforms a time series into a symbolic representation and extracts patterns that are used to generate trading signals: BUY, SELL, or HOLD.

---

## Description

The approach is based on the following steps:

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

---

## Functionality

The project includes:

* preprocessing of financial time series
* SAX discretization
* extraction of association rules
* walk-forward validation without data leakage
* generation of trading signals
* evaluation of prediction accuracy
* calculation of strategy returns

---

## Streamlit Application

The repository includes an interactive web application that allows:

* uploading a CSV file with time series data
* configuring model parameters
* viewing predictions and evaluation metrics
* exploring generated association rules

---

## Application Interface

### Main Screen
![Main](interface.png)

### Time Series Visualization
![Chart](chart.png)

### Prediction Metrics
![Metrics](metrics.png)

### Trading Signal
![Signal](recommendation.png)

### Association Rules
![Rules](rules.png)

## System Design

### Use Case Diagram
![Use Case Diagram](uml_use_case.png)

---

## Project Structure

```
.
├── app.py
├── main.py
├── elbow_method.py
├── README.md
```

---

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

* The application interface is in Russian
* The method can be applied to different types of time series
* The project is part of a diploma research work

---

## Author

Ekaterina Polosmak


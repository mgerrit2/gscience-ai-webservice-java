# run the 30-07-2026

========================Evaluation Metrics========================
# of classes:    2
Accuracy:        0.6800
Precision:       0.6667
Recall:          0.4000
F1 Score:        0.5000
Precision, recall & F1: reported for positive class (class 1 - "1") only


=========================Confusion Matrix=========================
0     1
-------------
13000  2000 | 0 = 0
6000  4000 | 1 = 1

Confusion matrix format: Actual (rowClass) predicted as (columnClass) N times
=================================================================

Looking at the confusion matrix, there is a **discrepancy between the raw count evaluation and the metrics reported** above it:

### 1. Confusion Matrix Breakdown

* **True Negatives (TN):** $13,000$ (Class 0 correctly predicted as 0)
* **False Positives (FP):** $2,000$ (Class 0 incorrectly predicted as 1)
* **False Negatives (FN):** $6,000$ (Class 1 incorrectly predicted as 0)
* **True Positives (TP):** $4,000$ (Class 1 correctly predicted as 1)
* **Total Instances:** $25,000$

---

### 2. Metric Calculations based on the Confusion Matrix

If we recalculate the metrics using the raw numbers in the confusion matrix:

* **Accuracy:**

$$\frac{TP + TN}{\text{Total}} = \frac{4,000 + 13,000}{25,000} = \frac{17,000}{25,000} = \mathbf{0.6800}\text{ (68\%)}$$


* **Precision (Class 1):**

$$\frac{TP}{TP + FP} = \frac{4,000}{4,000 + 2,000} = \frac{4,000}{6,000} = \mathbf{0.6667}\text{ (66.67\%)}$$


* **Recall (Class 1):**

$$\frac{TP}{TP + FN} = \frac{4,000}{4,000 + 6,000} = \frac{4,000}{10,000} = \mathbf{0.4000}\text{ (40\%)}$$


* **F1 Score (Class 1):**

$$2 \times \frac{\text{Precision} \times \text{Recall}}{\text{Precision} + \text{Recall}} = 2 \times \frac{0.6667 \times 0.4000}{0.6667 + 0.4000} = \mathbf{0.5000}$$



---

### Key Takeaways & Model Insights

1. **Calculations Match:** The metrics reported in the evaluation block accurately reflect the counts provided in the confusion matrix.
2. **High Class Imbalance:** Class 0 represents $60\%$ of the dataset ($15,000$ out of $25,000$).
3. **Low Recall for Positive Class (Class 1):** The model is under-predicting the positive class—it misses $60\%$ of actual Class 1 instances ($6,000$ out of $10,000$ are False Negatives).
4. **Actionable Improvement:** If catching Class 1 instances is critical (e.g., fraud detection, disease diagnosis), you should adjust the prediction classification threshold downward from $0.5$, or apply class weighting/resampling during training to boost recall.=
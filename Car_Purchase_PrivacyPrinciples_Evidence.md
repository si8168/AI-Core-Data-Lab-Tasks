# Privacy Engineering Evidence Document: Car Purchase Amount Prediction
**Course Module Assignment | Security & Privacy Principles Throughout the SDLC**

---

### 1. Introduction
The objective of this machine learning project is to build a regression model that estimates a customer's car purchase budget based on financial demographics. Because this system processes customer transaction data, embedding privacy principles into the Software Development Lifecycle (SDLC) is critical. Safeguarding consumer privacy preserves organizational data compliance, mitigates security leakage vectors, and builds operational trust by ensuring that Personally Identifiable Information (PII) is completely isolated from algorithmic execution loops.

---

### 2. Planning and Analysis
Through programmatic exploratory data analysis (`.head()`, `.info()`, `.columns.tolist()`), we scanned the raw database attributes to evaluate data-collection minimizations. 
* **Privacy Risks Identified:** The raw files contain tracking strings like `Customer Name`, `Customer e-mail`, and geographical indicators like `Country`. 
* **Impact Evaluation:** These elements do not contribute mathematical weight to statistical variance or purchase vectors. Leaving them in introduces data leakage surface areas, exposing sensitive credentials if model assets are shared.

---

### 3. Privacy Decision Table

| Column Name | Keep or Remove | Reason |
| :--- | :--- | :--- |
| **Customer Name** | Remove | Personal Identifier (PII). Not mathematically required for prediction. |
| **Customer e-mail**| Remove | Personal Identifier (PII). High risk for identity leakage. |
| **Country** | Remove | Unnecessary noise attribute for this localized core prediction engine. |
| **Gender** | Keep | Required input demographic feature vector. |
| **Age** | Keep | Required input demographic feature vector. |
| **Annual Salary** | Keep | Core continuous economic metric indicating purchase capacity. |
| **Credit Card Debt**| Keep | Core continuous economic metric indicating current liabilities. |
| **Net Worth** | Keep | Core continuous economic metric indicating available capitalization. |
| **Car Purchase Amount**| Keep | Target Output variable for Supervised Regression execution. |

---

### 4. Design Spec
* **Approved Input Features:** `Gender`, `Age`, `Annual Salary`, `Credit Card Debt`, `Net Worth`
* **Target Variable:** `Car Purchase Amount`
* **Sanitized Deletions:** `Customer Name`, `Customer e-mail`, `Country`

---

### 5. Development Phase Compliance
Our production cleaning functions actively block non-predictive variables using programmatic filters before any arrays interact with our model algorithms:
```python
columns_to_drop = ["Customer Name", "Customer e-mail", "Country"]
data_cleaned = data.drop(columns=columns_to_drop, axis=1)
X = data_cleaned.drop("Car Purchase Amount", axis=1)
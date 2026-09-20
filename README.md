# Bayesian Statistical Modelling & MCMC Diagnostics

A complete repository containing hands-on implementations of **Bayesian Data Analysis**, probabilistic programming with **PyMC**, and exploratory posterior analysis with **ArviZ**. 

This repository covers both **Generalized Linear Models (Gaussian & Poisson)** and **Hierarchical (Multilevel) Normal Models**, emphasizing parameter recovery, MCMC diagnostics ($\hat{R}$, ESS, divergences), non-centered parameterization, and partial pooling (shrinkage).

---

## 📌 Repository Overview

This repository consists of two main Jupyter Notebooks:

1. **`Bayesian Modeling (Linear & Poisson Regression).ipynb`**
   * **Problem 1:** Bayesian Linear Regression — Predicting exam scores from study hours using a Gaussian likelihood.
   * **Problem 2:** Bayesian Poisson Regression — Predicting hospital patient admissions from ambient temperature using a Poisson likelihood with a log-link function.
2. **`Hierarchical_(Multilevel)_Models.ipynb`**
   * **Scenario 1:** Hierarchical Normal Model — Modeling nested student performance data across $J = 8$ unbalanced school groups using non-centered parameterization to mitigate sampling funnels.

---

## 📐 Models Specification & Formulations

### 1. Bayesian Linear Regression (Gaussian Likelihood)
Predicting continuous target variables $y_i$ with normal observation noise:

$$
y_i \sim \text{Normal}(\mu_i, \sigma) \quad \text{where} \quad \mu_i = \alpha + \beta \cdot \text{StudyHours}_i
$$

* **Priors:**
  * $\alpha \sim \text{Normal}(50, 20)$
  * $\beta \sim \text{Normal}(5, 5)$
  * $\sigma \sim \text{HalfNormal}(10)$

---

### 2. Bayesian Poisson Regression (Log-Link Function)
Predicting count data $y_i \ge 0$ (e.g., daily hospital admissions) constrained by dynamic rate $\lambda_i$:

$$
y_i \sim \text{Poisson}(\lambda_i) \quad \text{where} \quad \log(\lambda_i) = \alpha + \beta \cdot \text{Temperature}_i
$$

* **Priors:**
  * $\alpha \sim \text{Normal}(3, 1)$
  * $\beta \sim \text{Normal}(0, 0.5)$

---

### 3. Hierarchical (Multilevel) Normal Model
Modeling nested scores $y_i$ belonging to school $j[i]$ with non-centered parameterization:

$$
y_i \sim \text{Normal}(\theta_{j[i]}, \sigma)
$$

Where group means $\theta_j$ are drawn from a shared hyper-distribution:

$$
\theta_{\text{offset}, j} \sim \text{Normal}(0, 1), \quad \theta_j = \mu + \theta_{\text{offset}, j} \cdot \tau
$$

* **Hyperpriors:**
  * Overall Mean ($\mu$): $\mu \sim \text{Normal}(75, 15)$
  * Group Variance ($\tau$): $\tau \sim \text{HalfNormal}(10)$
  * Individual Variance ($\sigma$): $\sigma \sim \text{HalfNormal}(15)$

---

## 🛠️ Tech Stack & Requirements

* **Python 3.9+**
* [**PyMC**](https://www.pymc.io/) — Probabilistic programming framework utilizing NUTS (No-U-Turn Sampler).
* [**ArviZ**](https://arviz-devs.github.io/arviz/) — Bayesian exploratory data analysis and convergence diagnostic visualization.
* **NumPy, Pandas, Matplotlib, Seaborn** — Data manipulation and plotting.

Install all dependencies via pip:

```bash
pip install pymc arviz numpy pandas matplotlib seaborn
```

---

## 📈 Results & Diagnostic Summaries

### 1. Linear & Poisson Regression Performance
* **Convergence:** All Markov chains achieved perfect convergence ($\hat{R} = 1.00$) with zero sampling divergences.
* **Posterior Predictive Checks (PPC):** PPC overlay plots confirm that the posterior predictive distributions closely match the empirical observations for both continuous exam scores and discrete hospital admission counts.

### 2. Hierarchical Model & Partial Pooling
* **Parameter Recovery:**
  * **District Mean ($\mu$):** True = $75.00$ | Estimated Posterior Mean = $77.63$ ($95\% \text{ HDI}: [74.50, 80.98]$)
  * **Between-School Variance ($\tau$):** True = $5.00$ | Estimated Posterior Mean = $4.34$ ($95\% \text{ HDI}: [1.89, 7.53]$)
  * **Within-School Variance ($\sigma$):** True = $8.00$ | Estimated Posterior Mean = $8.01$ ($95\% \text{ HDI}: [7.36, 8.58]$)
* **Shrinkage Demonstration:**
  * **Small Sample Size (School 4, $N=8$):** Sample mean $75.89$ shrinks towards the grand mean $\mu \approx 77.63$ ($\hat{\theta}_4 = 76.53$).
  * **Large Sample Size (School 3, $N=80$):** Sample mean $82.29$ resists shrinkage due to high data volume ($\hat{\theta}_3 = 82.04$).

---

## 🚀 How to Run

1. **Clone the Repository:**
   ```bash
   git clone https://github.com/neupra/Bayesian-Statistical-Modelling.git
   cd Bayesian-Statistical-Modelling
   ```

2. **Launch Jupyter Notebook / Google Colab:**
   ```bash
   jupyter notebook
   ```

3. **Execute Notebooks:**
   Run cells sequentially in both `Bayesian Modeling (Linear & Poisson Regression).ipynb` and `Hierarchical_(Multilevel)_Models.ipynb`.

---

## 💡 Key Takeaways
* **Non-Centered Parameterization** effectively breaks geometry funnels in hierarchical models, allowing smooth chain exploration even with small group sample sizes.
* **Partial Pooling** provides an optimal trade-off between complete pooling (ignoring group differences) and no-pooling (overfitting to small noisy samples).

# Bayesian Modeling (Linear & Poisson Regression) & MCMC Diagnostics

A comprehensive **Full Bayesian Modeling Pipeline** built in Python using **PyMC** and **ArviZ**. This repository demonstrates how to formulate, sample, evaluate, and compare Bayesian linear and count-based regression models with strict Markov Chain Monte Carlo (MCMC) diagnostics and Posterior Predictive Checks (PPC).

## 📌 Project Overview

This project tackles two distinct regression problems to showcase end-to-end Bayesian inference:

1. **Problem 1: Bayesian Linear Regression (Gaussian Likelihood)**

   * **Goal:** Predict exam scores derived from study hours.

   * **Model:** $y_i \sim \mathcal{N}(\mu_i, \sigma)$ with $\mu_i = \beta_0 + \beta_1 \cdot x_i$

   * **Focus:** Interpreting continuous parameters ($\beta_0$ intercept, $\beta_1$ slope, $\sigma$ residual standard deviation) and visualising posterior parameter uncertainty.

2. **Problem 2: Bayesian Poisson Regression (Poisson Likelihood + Log-Link)**

   * **Goal:** Predict hospital patient visits based on ambient temperature.

   * **Model:** $y_i \sim \text{Poisson}(\lambda_i)$ with $\log(\lambda_i) = \beta_0 + \beta_1 \cdot x_i$

   * **Focus:** Handling non-negative count data, exponential growth rates via log-link, and addressing potential overdispersion.

## 🛠️ Key Features & Workflow

For both Bayesian models, the pipeline performs:

* **Exploratory Data Analysis (EDA):** Tabular summaries and correlation/scatter visualisations.

* **Prior Specification & Justification:** Defining weakly informative Gaussian and Half-Normal priors.

* **MCMC Sampling:** Executing NUTS (No-U-Turn Sampler) via **PyMC** across multiple chains.

* **Convergence & MCMC Diagnostics:** Examining trace plots, rank plots, $\hat{R}$ statistics ($R$-hat), and Effective Sample Sizes ($ESS_{bulk}$, $ESS_{tail}$).

* **Posterior Inference & Interpretation:** Summarizing parameter expectations, standard deviations, and 94% High Density Intervals (HDI).

* **Posterior Predictive Checks (PPC):** Evaluating fit by sampling from the posterior predictive distribution vs. observed data.

* **Comparative Analysis:** A side-by-side comparison contrasting linear vs. Poisson likelihood assumptions.

##  Tech Stack & Environment

* **Python:** `^3.10`

* **PyMC:** `v5.x` (Probabilistic Programming Framework)

* **ArviZ:** `v0.x` (Exploration and diagnostics of Bayesian models)

* **NumPy & Pandas:** Data manipulation

* **Matplotlib & Seaborn:** Data and posterior visualisations

##  Getting Started

### Prerequisites

Ensure you have a Python environment set up (Conda, Pyenv, or Virtualenv).

### Installation

1. **Clone the repository:**

   ```
   git clone https://github.com/your-username/your-repo-name.git
   cd your-repo-name
   
   ```

2. **Install dependencies:**

   ```
   pip install pymc arviz numpy pandas matplotlib
   
   ```

3. **Launch Notebook:**

   ```
   jupyter notebook
   
   ```

   Open `Project1_Bayesian_Modeling.ipynb` (or your primary `.ipynb` file) and run all cells.

##  Summary of Results

* **Linear Model:** High linear correlation ($r \approx 0.997$) yielding tight posterior intervals for exam score growth per study hour.

* **Poisson Model:** Captures exponential count scaling effectively while ensuring non-negative rate predictions ($\lambda_i$).

# Bayesian Hierarchical (Multilevel) Models

A complete implementation and diagnostic analysis of a **Hierarchical Normal Model** using Python, **PyMC**, and **ArviZ**. This project demonstrates simulated data generation, non-centered Bayesian parameterization, MCMC sampling convergence diagnostics, parameter recovery, and partial pooling (shrinkage) analysis across nested group-level data.

---



---

## 📖 Project Overview

When modeling grouped or nested data (e.g., student test scores nested within different schools), simple linear models often suffer from either **underfitting** (complete pooling across all groups) or **overfitting** (no pooling, treating each group independently).

This project uses **Bayesian Hierarchical Modeling (Partial Pooling)** to allow individual group means to borrow strength from the overall population distribution. This is especially useful when sample sizes vary widely across groups.

---

## 📐 Model Specification

### Mathematical Formulation
For individual score $y_{i}$ belonging to school $j[i]$:

$$y_i \sim \text{Normal}(\theta_{j[i]}, \sigma)$$

Where group-level means $\theta_j$ are drawn from a shared hyper-distribution:

$$\theta_j \sim \text{Normal}(\mu, \tau)$$

With hyperpriors:
- **District Mean ($\mu$):** $\mu \sim \text{Normal}(75, 15)$
- **Between-Group Variance ($\tau$):** $\tau \sim \text{HalfNormal}(10)$
- **Within-Group Variance ($\sigma$):** $\sigma \sim \text{HalfNormal}(15)$

---

### Non-Centered Parameterization
To prevent sampling issues caused by "Neal's Funnel" geometry in hierarchical models with low group sample sizes, a **non-centered parameterization** is implemented:

$$\theta_{\text{offset}, j} \sim \text{Normal}(0, 1)$$

$$\theta_j = \mu + \theta_{\text{offset}, j} \cdot \tau$$

This uncouples the group-level parameters $\theta_j$ from the hyperparameter $\tau$ during MCMC sampling, significantly improving Markov chain exploration.

---

## 📊 Simulation & Dataset Details

Synthetic data was simulated across $J = 8$ schools with unbalanced sample sizes to observe partial pooling effects under varying group capacities:

* **Total Observations ($N$):** 280 students
* **Number of Groups ($J$):** 8 schools
* **Group Sample Sizes:** Ranges from $N_j = 8$ (School 4) to $N_j = 80$ (School 3)
* **True Parameters:**
  * $\mu_{\text{true}} = 75.0$
  * $\tau_{\text{true}} = 5.0$
  * $\sigma_{\text{true}} = 8.0$

---

## 🛠️ Installation & Requirements

Ensure you have Python 3.9+ installed along with the required libraries:

```bash
pip install pymc arviz numpy pandas matplotlib seaborn
```

### Key Libraries Used
* **[PyMC](https://www.pymc.io/):** Probabilistic programming framework for MCMC sampling (NUTS algorithm).
* **[ArviZ](https://arviz-devs.github.io/arviz/):** Exploratory analysis of Bayesian models, posterior plots, and diagnostic metrics.
* **Pandas & NumPy:** Data processing and synthetic generation.

---

## 📈 Results & Diagnostics

### MCMC Diagnostics
Sampling was performed with $4$ chains, $1,000$ tuning steps, and $1,500$ posterior draws per chain ($6,000$ total draws).

* **Convergence ($\hat{R}$):** $1.00$ across all parameters ($\mu, \tau, \sigma, \theta_0 \dots \theta_7$), confirming chain convergence.
* **Effective Sample Size ($\text{ESS}_{\text{bulk}}$):** High bulk ESS across all parameters ($\sigma \approx 2230$, $\theta_j \approx 2400-3500$, $\mu \approx 637$, $\tau \approx 611$).
* **Divergences:** 3 minor divergences detected during warmup/sampling.

---

### Parameter Recovery

Comparing true simulation parameters with posterior mean estimates and 95% High Density Intervals (HDI):

| Parameter | Description | True Value | Posterior Mean | 95% HDI ($3\% - 97\%$) | Status |
| :--- | :--- | :---: | :---: | :---: | :---: |
| **$\mu$** | Overall District Mean | **75.00** | **77.63** | $[74.50, 80.98]$ | ✅ Recovered |
| **$\tau$** | Between-School SD | **5.00** | **4.34** | $[1.89, 7.53]$ | ✅ Recovered |
| **$\sigma$** | Within-School SD | **8.00** | **8.01** | $[7.36, 8.58]$ | ✅ Accurate |

---

### Shrinkage & Partial Pooling

The hierarchical structure demonstrates clear shrinkage behavior based on group sample size:

1. **Small Sample Size (School 4, $N=8$):**
   * Sample Mean: `75.89`
   * Posterior Estimate ($\hat{\theta}_4$): `76.53`
   * *Observation:* Heavily pulled (shrunk) toward the overall mean $\mu \approx 77.63$ due to high uncertainty in small data.

2. **Large Sample Size (School 3, $N=80$):**
   * Sample Mean: `82.29`
   * Posterior Estimate ($\hat{\theta}_3$): `82.04`
   * *Observation:* Resists shrinkage; strong empirical data overrides the population prior.

---

## 🚀 How to Run

1. **Clone or Download the Repository:**
   ```bash
   git clone https://github.com/neupra/Bayesian-Statistical-Modelling
   cd bayesian-hierarchical-models
   ```

2. **Launch Jupyter Notebook:**
   ```bash
   jupyter notebook
   ```

3. **Open Notebook:**
   Open `Project_2_Bayesian.ipynb` and run all cells sequentially.

---

## 💡 Recommendations & Future Improvements

To further improve sampling efficiency and model presentation:
* **Eliminate Divergences:** Increase `target_accept` in `pm.sample()` from `0.80` to `0.95` or `0.99`:
  ```python
  idata_normal = pm.sample(
      draws=1500, 
      tune=1000, 
      target_accept=0.95, 
      random_seed=42
  )
  ```
* **Visualizations:** Add ArviZ forest plots (`az.plot_forest`) to visually compare raw sample means vs. posterior expectations across all schools.

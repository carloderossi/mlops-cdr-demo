# Azure ML Champion–Challenger MLOps Demo for Credit Risk Modeling

# Objective

Create a simplified end-to-end MLOps demonstration on Azure Machine Learning showcasing the Champion–Challenger deployment pattern using a Credit Risk prediction model.

The project focuses on demonstrating:

- Azure ML operational workflows
- model monitoring
- drift detection
- automated retraining triggers
- Champion/Challenger rollout strategies
- Azure Managed Online Endpoints

The implementation prioritizes simplicity, clarity, and reproducibility over banking-grade realism.

---

# Business Scenario

The system predicts whether a customer will default on a loan using a simplified binary classification Credit Risk model.

The project simulates a real-world scenario where:

- data distributions change over time
- model performance degrades
- retraining becomes necessary
- a challenger model must progressively replace the current production model

The demo also simulates delayed ground truth:
labels are assumed to become available only after some time.

For simplicity, delayed truth is simulated using pre-generated datasets that are revealed later in the workflow.

---

# Model Choice

## Algorithm

The recommended model is:

- XGBoost Classifier

Reasons:

- widely used in Credit Risk
- strong performance on tabular data
- interpretable feature importance
- simple to train and deploy
- suitable for demonstrating drift and retraining

---

# Dataset

## Recommended Dataset

Use a simplified tabular credit dataset such as:

- German Credit Dataset
- Home Credit Default Risk subset
- Synthetic generated credit data

The easiest option is likely:

- a synthetic dataset generated with Python using sklearn and pandas

Advantages:

- easy drift simulation
- no licensing concerns
- fully controllable
- small and lightweight

---

# Drift Simulation

To keep implementation simple, simulate:

## Covariate Drift

Example:

- increase average income
- change age distribution
- modify debt ratio ranges

This is:

- easy to generate
- easy to detect
- sufficient for demo purposes

No concept drift or IFRS9-style behavioral complexity is required.

---

# Economic Regime Change Simulation

To make the demo more realistic and business-oriented, the drift simulation should mimic a simplified economic downturn.

Example effects:

- lower average customer income
- higher debt-to-income ratio
- higher default probability
- reduced financial stability

This provides a more realistic explanation for:

- model decay
- retraining necessity
- challenger promotion

while remaining extremely simple to implement.

---

# Infrastructure

Infrastructure is provisioned using Bicep templates.

The environment includes:

- Azure ML Workspace
- Azure ML Registry
- compute resources
- storage
- Managed Online Endpoint
- monitoring resources

The existing Bicep repository is reused.

---

# ML Registry Assets

The Azure ML Registry stores reusable assets:

- YAML Conda environments
- datasets
- trained models

Training pipelines reference these assets directly.

---

# Training Workflow

The model is trained using:

- Azure ML Python Job
or
- Azure ML Pipeline

Outputs include:

- trained model artifact
- evaluation metrics
- model registration
- MLflow-compatible model packaging

The trained model is stored in the Azure ML Registry using MLflow model format.

Using MLflow format provides:

- standardized model packaging
- easier deployment and reproducibility
- compatibility with Azure ML model lifecycle management
- easier versioning and registry integration

---

# Evaluation Metric

Primary evaluation metric:

- ROC-AUC

Secondary optional metrics:

- F1 Score
- Precision
- Recall

ROC-AUC is preferred because:

- standard in Credit Risk
- threshold independent
- easy to explain

---

# Initial Deployment

The first trained model becomes:

## Champion Model (ModelA)

Deployment target:

- Azure Managed Online Endpoint

Inference requests are validated using a lightweight Python client.

---

# Inference Client

A lightweight Python client generates inference traffic.

Responsibilities:

- send prediction requests
- simulate production traffic
- generate monitoring logs

Authentication:

- client certificate
- Microsoft Entra ID application registration

---

# Monitoring

The system monitors:

- inference traffic
- prediction distributions
- data drift
- model performance

Preferred implementation:

- Azure ML Data Drift Monitor
- Azure Monitor
- Application Insights (optional)

The goal is operational simplicity aligned with Azure ML best practices.

---

# Drift Explanation Panel

After drift detection, the system should generate a lightweight explanation summary.

Example:

- average income decreased by 18%
- debt ratio increased by 22%
- default probability increased by 9%

Implementation:

- compare statistics between baseline and inference datasets
- display changes in the Streamlit dashboard

This significantly improves interpretability and presentation quality with minimal implementation effort.

---

# Delayed Ground Truth

Ground truth labels are simulated using pre-generated datasets.

Implementation approach:

- inference data is sent first
- labels are revealed later
- evaluation occurs after delayed label availability

This avoids implementing real delayed event systems while preserving the business concept.

---

# Retraining

When drift thresholds are exceeded:

- a GitHub Actions workflow is triggered
- retraining starts manually or semi-automatically

Retraining uses:

- historical stable data
- newly drifted data

This demonstrates adaptation while retaining historical patterns.

Rollback decisions remain fully manual.

---

# Human Approval Gate

Before promoting a Challenger model to production traffic, a manual approval step is required.

Implementation:

- GitHub Actions environment approval
or
- manual deployment confirmation

Purpose:

- simulate enterprise governance
- demonstrate human-in-the-loop operational control
- avoid fully automated production promotion

---

# Champion–Challenger Architecture

# Step 1 — Champion Deployment

ModelA is deployed as production Champion.

Traffic allocation:

- 100% → ModelA

---

# Step 2 — Challenger Training

After drift detection:

- ModelB is trained
- offline evaluation compares ModelA vs ModelB

If ModelB performs better, continue to shadow deployment.

---

# Step 3 — Shadow Deployment

ModelB is deployed alongside ModelA.

Behavior:

- ModelB receives mirrored traffic
- users still receive predictions only from ModelA
- predictions from both models are logged and compared

Traffic:

- 100% visible traffic → ModelA
- mirrored shadow traffic → ModelB

---

# Step 4 — Progressive Rollout

If ModelB consistently performs better:

Traffic routing progresses manually:

- 10% → ModelB
- 90% → ModelA

Then progressively:

- 25% / 75%
- 50% / 50%
- 75% / 25%
- 100% / 0%

Azure endpoint traffic rules are used for routing.

---

# Step 5 — Champion Replacement

When ModelB fully replaces ModelA:

- ModelB becomes the new Champion
- ModelA remains deployed temporarily as rollback fallback

Rollback remains manual.

---

# Model Version Lineage and Lifecycle Tracking

Each registered model should contain metadata tags such as:

- model version
- deployment date
- drift trigger reason
- evaluation metrics
- Champion/Challenger status
- rollout percentage

Example:

| Version | Status | ROC-AUC | Drift Trigger | Traffic |
|---|---|---|---|---|
| v1 | Champion | 0.81 | Initial Training | 90% |
| v2 | Challenger | 0.84 | Covariate Drift | 10% |

This provides:

- operational traceability
- model governance visibility
- easier dashboard visualization
- better storytelling during demos

---

# Explainability

Basic explainability is included using:

- XGBoost feature importance
or optionally:
- SHAP values

The goal is to demonstrate why tree-based models remain common in Credit Risk scenarios.

Full regulatory explainability is outside project scope.

---

# Feature Importance Comparison

The dashboard should compare feature importance between:

- Champion model
- Challenger model

Example:

| Feature | Champion Importance | Challenger Importance |
|---|---|---|
| Income | 0.42 | 0.25 |
| Debt Ratio | 0.18 | 0.39 |

This demonstrates:

- explainability
- impact of drifted data
- how the model adapts to changing economic conditions

This feature provides strong visual impact with minimal implementation complexity.

---

# Scope Exclusions

The project intentionally excludes:

- IFRS9 compliance
- PD/LGD/EAD multi-model frameworks
- regulatory governance
- multi-environment deployment
- automated rollback
- advanced feature stores
- production-grade MRM controls

These can later be discussed in a follow-up article.

---

# Bonus GUI

Preferred implementation:

## Streamlit Dashboard

Reasons:

- easiest to implement
- integrates well with Python
- sufficient for demo visualization

Dashboard displays:

- Champion vs Challenger metrics
- traffic allocation
- drift indicators
- rollout stage
- ROC-AUC comparison
- prediction distribution comparison
- feature importance comparison
- model lifecycle history
- drift explanation summary
- operational event timeline

---

# Operational Timeline Panel

The Streamlit dashboard should contain a simple operational event timeline.

Example:

| Timestamp | Event |
|---|---|
| 09:00 | Drift detected |
| 09:05 | Retraining triggered |
| 09:20 | Challenger deployed |
| 10:00 | 10% rollout started |
| 12:00 | Challenger promoted |

Purpose:

- improve operational storytelling
- simulate real-world monitoring workflows
- make the demo appear closer to a production AI platform

Implementation can be done using a simple dataframe or event log.

---

# GitHub Repository Contents

## Infrastructure

- Bicep templates

## ML Assets

- YAML Conda environments
- training datasets
- inference datasets
- retraining datasets

## ML Code

- training scripts
- evaluation scripts
- deployment scripts
- drift simulation scripts

## Automation

- GitHub Actions workflows
- retraining workflows
- manual approval gate

## Client

- inference client
- certificate authentication example

## Monitoring

- drift monitoring configuration
- Azure ML monitoring setup

## Dashboard

- Streamlit GUI

## Documentation

- architecture diagrams
- deployment instructions
- operational workflow explanation
- Champion–Challenger lifecycle description
- model governance explanation
- simplified explanation of delayed ground truth
- explanation of why tree models are commonly used in Credit Risk


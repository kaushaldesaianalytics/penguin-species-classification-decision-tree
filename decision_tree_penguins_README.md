# Penguin Species Classification
## Decision Tree Classifier

Which physical measurements best predict penguin species? This project builds and interprets a Decision Tree classifier on the Palmer Penguins dataset, progressing from an unconstrained baseline tree through hyperparameter pruning to find the simplest model that maintains strong accuracy.

---

## Overview

Decision trees partition the feature space through a sequence of binary splits, choosing at each node the feature and threshold that best separates the classes. Unlike black-box models, the resulting tree can be fully visualized and interpreted: every prediction traces a path through explicit biological rules such as body mass thresholds or culmen depth ranges.

This project demonstrates the full Decision Tree workflow including feature engineering, baseline modeling, tree visualization, feature importance analysis, and hyperparameter exploration across max depth, max leaf nodes, and split criterion.

---

## Dataset

The [Palmer Penguins dataset](https://allisonhorst.github.io/palmerpenguins/) contains 344 records across three species collected from three islands in the Palmer Archipelago, Antarctica. After dropping 10 records with missing values (2.9%), 334 clean records remain.

| Feature | Description |
|---|---|
| culmen_length_mm | Length of the culmen (bill ridge) in mm |
| culmen_depth_mm | Depth of the culmen in mm |
| flipper_length_mm | Flipper length in mm |
| body_mass_g | Body mass in grams |
| island | Island of observation (Biscoe, Dream, Torgersen) |
| sex | Biological sex (MALE, FEMALE) |
| **species** | **Target: Adelie, Chinstrap, or Gentoo** |

---

## Workflow

**1. Exploratory Data Analysis**
A missing value audit finds 10 incomplete records across culmen, flipper, body mass, and sex columns. These are dropped rather than imputed given the dataset size. Pairplots and scatter plots confirm strong visual separation between species, particularly in body mass and culmen dimensions.

**2. Feature Engineering**
The `island` and `sex` columns are nominal categorical variables converted to binary dummy columns via one-hot encoding. `drop_first=True` removes one dummy per variable to avoid perfect multicollinearity.

**3. Baseline Decision Tree**
An unconstrained `DecisionTreeClassifier` grows until all leaves are pure. This produces a complex tree that fits the training data exactly but risks overfitting. The baseline establishes the performance ceiling before pruning.

**4. Tree Visualization**
The full learned tree is rendered using `plot_tree`, showing each node's split condition, Gini impurity, sample count, and class distribution. Colored fills indicate the majority class at each node, making the decision path easy to follow.

**5. Feature Importances**
Gini-based importance scores measure each feature's contribution to impurity reduction across all splits. Body mass is identified as the most important single predictor, followed by culmen length and flipper length.

**6. Hyperparameter Exploration**
Three pruning strategies are compared using a consistent `report_model` evaluation function:

- `max_depth=2`: limits tree depth to the two most discriminative splits
- `max_leaf_nodes=3`: caps total terminal nodes while allowing variable depth
- `criterion='entropy'`: switches from Gini impurity to information gain as the split quality metric

---

## Results

| Model | Weighted F1 |
|---|---|
| Baseline (unconstrained) | ~0.97 |
| Max depth = 2 | ~0.97 |
| Max leaf nodes = 3 | ~0.93 |
| Entropy criterion | ~0.97 |

The depth-2 model matches baseline accuracy with a fraction of the complexity, producing a three-level tree that is fully interpretable. Limiting leaf nodes to 3 forces the tree to reduce to two species before adding the third, causing a slight accuracy drop.

---

## Key Concepts

**Gini Impurity vs. Entropy:** Gini measures the probability of misclassifying a randomly drawn sample. Entropy measures information gain. Both guide the same search for the best split and typically produce similar trees, though entropy can occasionally identify different optimal thresholds.

**Overfitting and Pruning:** An unconstrained tree memorizes training data perfectly, including noise. Max depth and max leaf node constraints limit tree complexity, trading some training accuracy for better generalization to unseen data.

**Feature Importance Aggregation:** A single tree's importance scores can be unstable. Random Forest (see companion project) aggregates importances across many trees for a more reliable estimate.

---

## Stack

- Python 3
- Pandas, NumPy
- Matplotlib, Seaborn
- scikit-learn (DecisionTreeClassifier, plot_tree, ConfusionMatrixDisplay, classification_report)

---

## File Structure

```
decision-tree-penguins/
├── decision_tree_penguins.ipynb   # Main project notebook
├── penguins_size.csv              # Palmer Penguins dataset
└── README.md
```

---

## How to Run

1. Clone the repository
2. Install dependencies: `pip install pandas numpy matplotlib seaborn scikit-learn`
3. Open `decision_tree_penguins.ipynb` in Jupyter and run all cells

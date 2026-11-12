# Isolation Forest experiment summary: B1–B22

## Decision

Use **B6b (20 features)** as the working final Isolation Forest configuration, while preserving the 23-feature version as a comparison baseline. B6b removes these three potentially capped count features:

- `Sent tnx`
- `Received Tnx`
- `total transactions (including tnx to create contract)`

Their caps are suspected from observed endpoint concentrations, but cannot be confirmed without the original data-collection/preprocessing recipe.

## Key evidence from experiments

- **B6 seed/stability comparisons:** B6b had the highest pairwise top-5% Jaccard among A, B1, B6a and B6b (0.8689) in that experiment.
- **B7 held-out ranking:** B6b and the 23-feature baseline shared 88 of 100 top-ranked held-out rows; their full rank correlation was 0.96837.
- **B8 matched-run comparisons:** A and B6b shared a mean 89.40 of 100 top-5% rows, with mean Jaccard 0.80923.
- **B9 scaling:** B6b and the robust-scaled equivalent had mean Spearman 0.99987 and top-5% Jaccard 0.97566; scaling had negligible impact.
- **B10 single-feature permutation:** `min value received` produced the largest mean Spearman drop (0.05264); `current_eth_balance` had the lowest mean top-5% Jaccard after permutation (0.74659).
- **B11 grouped permutation:** the `ERC20 uniq rec addr` / `ERC20 uniq rec token name` group had the strongest grouped sensitivity (mean Spearman drop 0.12533; mean top-5% Jaccard 0.65946).
- **B12 group ablation:** removing all four correlated groups left 12 features but substantially changed the shortlist (mean 59.04 of 100 top-5% rows shared with B6b; mean Jaccard 0.41929). Runtime improvement was negligible.
- **B13 alert budgets:** B6b had highest mean seed Jaccard at 1%, 5% and 10% alert budgets; B12 was highest at 2%. These measure ranking stability, not fraud yield.
- **B14–B15 consensus:** B14 reported 117 B6b rows with a top-5% appearance rate >=80% under its repeated held-out protocol. B15 found 94 rows meeting the same criterion in all three feature configurations (23-, 20-, and 18-feature models).
- **B16 duplicates and profiles:** none of the 94 consensus rows appeared in an exact duplicate feature group. Candidate medians were elevated for balance, ERC-20 received-token diversity, average sent values, average received values, and unique received addresses.
- **B17 feature tails:** highest frequency above background P95 included `max value received` (60/94), `total ether received` (59/94), `max val sent` (55/94), and `total Ether sent` (55/94).
- **B18–B20 exploratory clusters:** K-means with RobustScaler gave k=3 silhouette 0.67366 and sizes 77, 14, and 3. The same partitions were reported across B6b/B12/B9a representations and seeds; however, B22 shows that partitions are sensitive to transformation method. The three-cluster interpretation therefore remains exploratory.
- **B21 clustering sensitivity:** removing `min value received` changed the reference K-means clustering sharply (ARI 0.11011 for k=2 and mean ARI 0.14439 for k=3), but this is a statement about clustering the selected candidates, not about Isolation Forest fraud-detection quality.
- **B22 transformation sensitivity:** robust scaling and winsorised robust scaling retained the reference clustering fairly closely, whereas StandardScaler, QuantileTransformer and Yeo–Johnson changed cluster geometry substantially. These transformations were investigated for candidate clustering, not recommended for Isolation Forest.

## Limitations

1. The source dataset has 10,000 rows × 23 numeric features and no wallet identifier or verified fraud labels.
2. No precision, recall, F1, ROC-AUC, PR-AUC, fraud yield, or confirmed-fraud claims are supported by this dataset.
3. The consensus threshold (top-5% membership >=80% of repeated held-out evaluations) was exploratory. Repeated evaluations overlap and are not independent samples.
4. Marginal feature explanations show relative feature rarity, not exact tree-level causal explanations.
5. Several count features have heavy endpoint concentration. Capping/clipping is only a hypothesis until the upstream recipe is verified.
6. The clustering subtypes are descriptive and transformation-sensitive, not proven fraud categories.

## Recommended reporting language

“The selected B6b Isolation Forest provides a reproducible anomaly-ranking component. Across the conducted seed, feature-set and held-out sensitivity experiments, its ranking behaviour was reasonably consistent relative to the tested variants. Repeated evaluation identified 94 row-level candidates that met an exploratory consensus criterion across three feature configurations. Because the dataset contains neither confirmed labels nor wallet identifiers, these findings indicate anomalous behaviour in the supplied feature space; they do not validate fraud detection accuracy or identify real wallet addresses.”

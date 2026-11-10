# Cyberbullying Detection on Formspring Q&A Posts

A machine-learning project that classifies short social-media posts as **cyberbullying** or **not cyberbullying**, using TF-IDF text features and classical models (SVC, Naive Bayes, Decision Tree, Logistic/Ridge Regression, SVR).

> **Short version:** the models reach ~95% accuracy, but because only 6.5% of posts are bullying, accuracy is misleading. The real challenge, and the main lesson of this project, is detecting the rare bullying posts (best bullying-class F1 ≈ 0.42).

---

## Why this project?

**The problem.** Cyberbullying is a serious issue on social platforms, especially among young people. Platforms receive far more content than human moderators can review, so automated tools are needed to flag harmful posts for review.

**Why it is hard.**
- **Informal language:** slang, misspellings, stretched words (`uuuu`, `yuh`, `wht`), emoticons and HTML artefacts.
- **Severe class imbalance:** only about 1 in 15 posts is bullying. A model that always answers "not bullying" scores ~93.8% accuracy while detecting nothing.
- **Short text:** a few words per post give very little context.

**What I wanted to learn / show.**
1. Build an end-to-end NLP classification workflow: cleaning → TF-IDF → model tuning → evaluation.
2. See how different classical models behave on an imbalanced text problem.
3. Understand why **precision, recall and F1 on the minority class** matter more than accuracy.

---

## What does the project do?

The notebook `cyber_bullying_1.ipynb` walks through the full pipeline, with an *Insight* note after every code cell and a final conclusion.

| Step | Description |
|---|---|
| 1. Load & inspect | Read `Formspring.csv` (13,147 posts), check types, nulls and duplicates (3 duplicates removed → 13,144) |
| 2. Explore target | Count `Yes` / `No` labels and identify the class imbalance |
| 3. Encode labels | `Yes → 1` (bullying), `No → 0` |
| 4. Baseline | TF-IDF on **raw** text → SVC, Multinomial NB, Decision Tree |
| 5. Clean text | Remove punctuation and stop-words, apply Snowball stemming |
| 6. Vectorise | TF-IDF on the cleaned text (20,771 features) |
| 7. Split | 70 / 30 train-test split (9,200 / 3,944 posts), `random_state=4861` |
| 8. Tune | `GridSearchCV` for Ridge, Logistic Regression, SVR, RidgeClassifier, Multinomial NB and SVC |
| 9. Evaluate | Accuracy, precision, recall, F1, confusion matrices and classification reports |
| 10. Visualise | Bar charts comparing the metrics for each model family |

---

## Dataset

- **File:** `Formspring.csv` (13,147 rows × 2 columns)
- **Source:** Q&A posts from Formspring, a question-and-answer social network; this dataset is commonly used in cyberbullying-detection research (originally published by Reynolds et al., 2011).
- **Columns:**
  - `text`: the question and the reply combined, e.g. `Q: … <br>A: …`
  - `answer`: `Yes` if the post was labelled as cyberbullying, `No` otherwise
- **Class balance:** `No` = 12,292 (93.5%) vs `Yes` = 852 (6.5%) after de-duplication, roughly **14 : 1**.

---

## Results

### Baseline models (raw TF-IDF)

![Metric visualisation of the baseline models](Workshop_2_-_Graph.jpg)

*Scores in the chart are truncated to whole numbers.*

### All models, bullying class (class 1) on the 3,944-post test set

| Model | Features | Accuracy | Precision | Recall | F1 |
|---|---|---|---|---|---|
| **SVC (C≈3.16, RBF as run)** | cleaned | 94.98% | 73.7% | 29.8% | **42.4%** |
| Decision Tree | raw | 93.26% | 44.7% | **36.3%** | 40.1% |
| SVC (sigmoid, γ=1) | raw | **95.06%** | 82.9% | 25.7% | 39.3% |
| Logistic Regression (l1) | cleaned | 94.95% | 81.1% | 24.5% | 37.6% |
| SVR (sigmoid, rounded) | cleaned | 94.95% | 88.3% | 21.6% | 34.8% |
| Ridge (rounded) | cleaned | 94.90% | **89.3%** | 20.4% | 33.2% |
| Multinomial NB (α=0.2) | raw | 94.04% | 56.9% | 16.7% | 25.9% |
| Multinomial NB (α=1) | cleaned | 93.81% | 100% (1 flag) | 0.4% | 0.8% |

*Majority-class baseline ("always No"): 93.79% accuracy.*

### Key takeaways

- **Accuracy hides the real picture.** Every sensible model scores 94–95%, only ~1 point above the "always No" baseline.
- **Best balance:** the tuned-feature **SVC** (F1 0.42, ~74% precision, ~30% recall).
- **Highest recall:** the **Decision Tree** (36%), but at a precision of only 45%.
- **Naive Bayes with α=1 collapses to "always No"**, while α=0.2 behaved much better.
- **Overall:** the models are *high-precision, low-recall*. When they flag a post they are usually right, but they miss roughly 70–80% of bullying posts, so they are not yet suitable for real moderation.

---

## Known limitations

- The train/test split is **not stratified**, and TF-IDF is fitted on the full corpus before splitting (a mild leak).
- Tuning used accuracy, R² or weighted-F1 scoring, which favours the majority class.
- A few final models were rebuilt with parameters different from the tuned ones (Logistic Regression `C`/`tol`, SVC kernel), and the "Ridge Classifier" run is actually `Ridge`.
- The Naive Bayes `alpha` grid started at 1, so smaller values were never tested.
- Cleaning removes punctuation *before* decoding HTML, leaving artefact tokens such as `bra` and `039`.
- The **SGD Classifier and Regressor** grid searches were not run because they take too much time.

---

## Possible next steps

1. Use a stratified split with cross-validation, and put TF-IDF inside a `Pipeline`.
2. Use `class_weight="balanced"` (or resample the training data only) and tune on bullying-class F1 or PR-AUC.
3. Tune the decision threshold from the precision-recall curve to trade some precision for recall.
4. Improve cleaning: unescape HTML → remove `<br>` and `Q:` / `A:` markers → lowercase → tokenise → stop-words → stem.
5. Add character n-gram TF-IDF for slang and misspellings; try `LinearSVC` and `ComplementNB`.
6. Test the unused `Length` feature, then try a fine-tuned transformer (e.g. DistilBERT).

---

## Tech stack

Python · pandas · NumPy · NLTK · scikit-learn · Matplotlib · Jupyter Notebook

---

## How to run

```bash
# 1. Clone the repository
git clone <your-repo-url>
cd <your-repo-folder>

# 2. Install dependencies
pip install pandas numpy nltk scikit-learn matplotlib jupyter

# 3. Download the NLTK stop-words (one time)
python -c "import nltk; nltk.download('stopwords')"

# 4. Launch the notebook
jupyter notebook cyber_bullying.ipynb
```

Keep `Formspring.csv` in the same folder as the notebook. The grid searches (especially SVR and SVC) can take a long time to run.

---

## Repository structure

```
├── cyber_bullying_1.ipynb        # Full analysis with insights and conclusion
├── Formspring.csv                # Dataset
├── Workshop_2_-_Graph.jpg        # Baseline model comparison chart
└── README.md
```

---

## Author

**Ram Vilas Hasanapuram**: Aspiring Data Scientist (Python, SQL, Machine Learning)

- GitHub: [github.com/ramvilas273](https://github.com/ramvilas273)
- LinkedIn: [linkedin.com/in/ram-vilas](https://linkedin.com/in/ram-vilas)

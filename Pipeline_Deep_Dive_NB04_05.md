# Deep Dive: Transaction Foundation Model Pipeline
## Part 3: Embedding Extraction & Downstream Fraud Detection (Notebooks 04 & 05)

This document provides a detailed breakdown of the final stages of the pipeline, where the pre-trained foundation model is utilized to enhance a downstream XGBoost fraud detection model.

---

### 1. Notebook 04: Embedding Extraction

In Notebook 03, we trained a Foundation Model to learn the patterns of normal financial behavior. In Notebook 04, we use this model in "Inference Mode" to generate numerical representations (embeddings) for every single transaction.

#### The Process: From Tokens to Embeddings
During training (Notebook 03), the model learned the "rules of the game." By predicting the next token across millions of transactions, the neural network adjusted its internal weights to understand the grammar of human financial behavior. The output of training was a set of learned parameters (the `.safetensors` model weights), **not** embeddings for specific transactions.

In Notebook 04, we extract the embeddings by following these steps:
1. **GPU-Accelerated Tokenization**: We take our raw transaction data and tokenize it again using the exact same custom 12-step `FinancialTabularTokenizer`. These tokens act as the *input* for our extraction phase.
2. **Forward Pass (Inference Mode)**: We load the *frozen* learned weights from Notebook 03. We pass our tokenized transactions through this frozen model. Crucially, we do *not* ask the model to predict the next token or update its weights anymore. Instead, we are looking at the model's internal brain as it reads the data.
3. **Last-Token Pooling**: As the causal LLaMA model reads a transaction (which consists of 12 tokens, processed left-to-right), it calculates an internal "hidden state" at each step. To get a single numerical representation for the entire transaction, we grab the 512-dimensional hidden state at the *12th token* (the last non-padding token) of that transaction. Because it's a causal model, this 12th token has mathematically "attended" to all 11 previous tokens in the transaction, plus the user's entire historical sequence up to that point.
4. **The Output**: This 512-dimensional hidden state is our **embedding**. We extract and save this dense vector for every transaction. It represents the model's contextual "understanding" of that specific transaction.

#### Q&A: Understanding the Mechanics

**Q: We used the tabular tokens from NB2 to train the foundation model, and now we use the same tokens again in NB4? Why?**
**A:** Yes! In training (NB03), the tokens are used as both the *input* and the *target* to teach the model how to adjust its neural network weights. But training doesn't output embeddings; it outputs a "smart" model. To actually get the embeddings (the numerical representations of the transactions to give to XGBoost), we have to feed the tokens back into this "smart" model. We use the tokens as input to trigger the model's internal activations, and we capture those activations as our embeddings. 

**Q: Why reduce the embeddings from 512 dimensions to 64 dimensions with PCA? Why not just use the 512d embeddings directly in XGBoost?**
**A:** 512 features is a very high-dimensional space for a tree-based model like XGBoost, especially when our original tabular data only had 13 features. Feeding 512 dimensions directly can lead to severe overfitting (the model memorizes noise instead of learning the fraud pattern) and drastically slows down training. PCA acts as a noise filter, condensing the most important contextual information into 64 dimensions while dropping the less relevant variance.

#### Data Splits
To ensure a fair and apples-to-apples comparison later:
- **Train**: A balanced sample of 1 Million transactions (~2.5% fraud).
- **Validation/Test**: The exact same 100K stratified subsets generated back in Notebook 01 (~0.1% fraud).

---

### 2. Notebook 05: XGBoost Fraud Detection

This is where the actual Fraud Detection happens. We compare three different XGBoost models to see if the complex foundation model embeddings actually provide a tangible benefit over standard tabular modeling.

#### Dimensionality Reduction (PCA)
Before training, the 512-dimensional embeddings are reduced to **64 dimensions** using PCA (Principal Component Analysis). 
- *Why?* 512 dimensions contain a lot of noise. PCA retains the most important variance (~78%) while shrinking the feature space, which prevents the XGBoost model from severely overfitting.

#### The Three Contenders
Three XGBoost models are trained. Each undergoes rigorous Hyperparameter Optimization (HPO) to maximize its validation Average Precision (AP):

1. **Baseline Model (13d)**: Uses only the 13 raw tabular features (Amount, Merchant, MCC, Zip, Hour, etc.). This represents standard, traditional tabular modeling.
2. **Embeddings-Only Model (64d)**: Uses *only* the 64d PCA foundation model embeddings. No raw tabular data is provided.
3. **Combined Model (77d)**: Uses both the 13 raw features + the 64d PCA embeddings.

#### Results & Key Insights (Test Set Holdout)

| Model | Features | ROC-AUC | Average Precision (AP) |
| :--- | :---: | :---: | :---: |
| Baseline | 13d | 0.9885 | 0.1238 |
| Embeddings-Only | 64d | 0.8822 | 0.0168 |
| **Combined** | **77d** | **0.9906** | **0.1418** |

**Key Takeaways:**
1. **Embeddings alone are not enough**: The foundation model embeddings perform poorly on their own for fraud detection. This is because they were trained *without fraud labels*—they represent general behavioral context, not necessarily a hard fraud boundary.
2. **The "Combined" Model wins**: Adding the embeddings to the raw features yields the best performance. It gives the XGBoost model the best of both worlds: explicit hard features (like raw transaction amounts) *plus* deep, contextual behavioral history (the embeddings).
3. **Massive lift in Average Precision (AP)**: While the ROC-AUC lift seems small (+0.21%), the lift in Average Precision is massive (**+14.5%**). 
   - *Why does AP matter?* In the real world, fraud detection teams have a limited budget to review flagged transactions. Average Precision specifically measures the accuracy of the model's *most confident* predictions. A 14.5% lift in AP means the review team will encounter significantly fewer "false alarms" (False Positives) at the top of their queue, saving immense time and money.

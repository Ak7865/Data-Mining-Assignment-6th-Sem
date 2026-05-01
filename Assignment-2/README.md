# 📧 Assignment 2 — Spam Detection (Naive Bayes)

## 🎯 Objective
Build a **Spam Classifier** using the **Naive Bayes algorithm** *without using any machine learning libraries*.

---

## 📂 Dataset

**File:** `dataset.csv`

### Format:
| word  | not_spam | spam |
|------|----------|------|
| hello | 10       | 2    |
| offer | 1        | 15   |

---

## ⚙️ Implementation Logic

### 1️⃣ Load Dataset
- Read CSV file using Python `csv` module  
- Store word frequencies for both classes:
  - Spam
  - Not Spam  

---

### 2️⃣ Calculate Probabilities
Compute prior probabilities:

```
P(Not Spam)
P(Spam)
```

---

### 3️⃣ Classification Logic

Apply **Laplace Smoothing**:

```
P(word | class) = (count + 1) / (total_words + vocab_size)
```

---

### 4️⃣ Naive Bayes Formula

```
P(C | X) = (P(X | C) * P(C)) / P(X)
```

Where:
- `C` = Class (Spam / Not Spam)
- `X` = Input message

---

### 5️⃣ Prediction
- Calculate probabilities for both classes  
- Compare results  
- Output classification:

```
SPAM
or
NOT SPAM
```

---

## 📸 Output Screenshot

Add your output screenshot in the repository:

```
output.png
```

---

## 💻 Sample Output

[!Output]()
---

## 🛠️ Features

- ✅ No external ML libraries  
- ✅ Lightweight implementation  
- ✅ Beginner-friendly  
- ✅ Based on probability concepts  

---

## ⚠️ Limitations

- Assumes **word independence**
- Accuracy depends on dataset quality
- Not suitable for very large datasets

---

## 🚀 Future Improvements

- Add text preprocessing (stopwords, stemming)
- Use TF-IDF
- Build web interface
- Compare multiple algorithms

---

## 👨‍💻 Author

**Syed Akhter Hussain**  
B.Tech CSE Student  
Barak Valley Engineering College  

---
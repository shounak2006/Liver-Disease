import matplotlib.pyplot as plt
import seaborn as sns
import os

os.makedirs("results", exist_ok=True)

# -------- Confusion Matrix: Logistic Regression --------
cm_logistic = [[7, 27],
               [4, 79]]

plt.figure(figsize=(5,4))
sns.heatmap(cm_logistic, annot=True, fmt="d", cmap="Blues",
            xticklabels=["No Disease", "Disease"],
            yticklabels=["No Disease", "Disease"])
plt.xlabel("Predicted")
plt.ylabel("Actual")
plt.title("Confusion Matrix - Logistic Regression")
plt.tight_layout()
plt.savefig("results/confusion_matrix_logistic.png", dpi=300)
plt.close()

# -------- Confusion Matrix: Random Forest --------
cm_rf = [[9, 25],
         [8, 75]]

plt.figure(figsize=(5,4))
sns.heatmap(cm_rf, annot=True, fmt="d", cmap="Greens",
            xticklabels=["No Disease", "Disease"],
            yticklabels=["No Disease", "Disease"])
plt.xlabel("Predicted")
plt.ylabel("Actual")
plt.title("Confusion Matrix - Random Forest")
plt.tight_layout()
plt.savefig("results/confusion_matrix_rf.png", dpi=300)
plt.close()

# -------- Accuracy Comparison --------
models = ["Logistic Regression", "Random Forest", "Balanced Logistic", "SVM"]
accuracies = [0.735, 0.718, 0.726, 0.692]

plt.figure(figsize=(7,4))
plt.bar(models, accuracies)
plt.ylabel("Accuracy")
plt.ylim(0.6, 0.8)
plt.title("Model Accuracy Comparison")
plt.xticks(rotation=20)
plt.tight_layout()
plt.savefig("results/accuracy_comparison.png", dpi=300)
plt.close()

# -------- Recall Comparison --------
recall = [0.95, 0.90, 0.66, 0.63]

plt.figure(figsize=(7,4))
plt.bar(models, recall)
plt.ylabel("Recall (Liver Disease)")
plt.ylim(0.5, 1.0)
plt.title("Recall Comparison for Liver Disease Detection")
plt.xticks(rotation=20)
plt.tight_layout()
plt.savefig("results/recall_comparison.png", dpi=300)
plt.close()

print("All result images generated in /results folder")

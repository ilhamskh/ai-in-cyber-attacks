from sklearn.metrics import classification_report, confusion_matrix, accuracy_score, roc_curve, auc
import matplotlib.pyplot as plt
import seaborn as sns
import pandas as pd
import os

def evaluate_model(model, X_test, y_test, output_dir='results'):
    """
    Modelin performansını qiymətləndirir və nəticələri yadda saxlayır.
    """
    if not os.path.exists(output_dir):
        os.makedirs(output_dir)

    # Proqnozlar
    y_pred = model.predict(X_test)
    y_pred_proba = model.predict_proba(X_test)[:, 1]

    # Dəqiqlik
    accuracy = accuracy_score(y_test, y_pred)
    print(f"\nModel dəqiqliyi: {accuracy*100:.2f}%")

    # Confusion Matrix
    cm = confusion_matrix(y_test, y_pred)
    print("\nConfusion Matrix:")
    print(cm)

    # Hesabat
    print("\nKlassifikasiya hesabatı:")
    print(classification_report(y_test, y_pred, target_names=['Normal', 'Attack']))

    # Metrikalar
    tn, fp, fn, tp = cm.ravel()
    tpr = tp / (tp + fn)
    fpr = fp / (fp + tn)
    precision = tp / (tp + fp) if (tp + fp) > 0 else 0
    f1 = 2 * (precision * tpr) / (precision + tpr) if (precision + tpr) > 0 else 0

    print(f"\nƏlavə metrikalar:")
    print(f"True Positive Rate (TPR): {tpr*100:.2f}%")
    print(f"False Positive Rate (FPR): {fpr*100:.2f}%")
    print(f"Precision: {precision*100:.2f}%")
    print(f"F1-Score: {f1*100:.2f}%")

    # Plot Confusion Matrix
    plt.figure(figsize=(8, 6))
    sns.heatmap(cm, annot=True, fmt='d', cmap='Blues', 
                xticklabels=['Normal', 'Attack'],
                yticklabels=['Normal', 'Attack'])
    plt.title('Confusion Matrix')
    plt.ylabel('Həqiqi dəyər')
    plt.xlabel('Proqnozlaşdırılan dəyər')
    plt.savefig(os.path.join(output_dir, 'confusion_matrix.png'), dpi=300, bbox_inches='tight')
    plt.close()

    # ROC Curve
    fpr_roc, tpr_roc, thresholds = roc_curve(y_test, y_pred_proba)
    roc_auc = auc(fpr_roc, tpr_roc)

    plt.figure(figsize=(8, 6))
    plt.plot(fpr_roc, tpr_roc, color='darkorange', lw=2,
             label=f'ROC curve (AUC = {roc_auc:.3f})')
    plt.plot([0, 1], [0, 1], color='navy', lw=2, linestyle='--')
    plt.xlim([0.0, 1.0])
    plt.ylim([0.0, 1.05])
    plt.xlabel('False Positive Rate')
    plt.ylabel('True Positive Rate')
    plt.title('ROC Curve - Kiberhücum Aşkarlama')
    plt.legend(loc="lower right")
    plt.savefig(os.path.join(output_dir, 'roc_curve.png'), dpi=300, bbox_inches='tight')
    plt.close()

    return {
        'accuracy': accuracy,
        'tpr': tpr,
        'fpr': fpr,
        'precision': precision,
        'f1': f1,
        'roc_auc': roc_auc,
        'y_pred_proba': y_pred_proba
    }

def plot_feature_importance(model, feature_names, output_dir='results'):
    """
    Xüsusiyyətlərin əhəmiyyətini vizuallaşdırır.
    """
    feature_importance = pd.DataFrame({
        'feature': feature_names,
        'importance': model.feature_importances_
    }).sort_values('importance', ascending=False)

    plt.figure(figsize=(10, 6))
    sns.barplot(data=feature_importance.head(15), 
                x='importance', y='feature', palette='viridis')
    plt.title('Ən Əhəmiyyətli 15 Xüsusiyyət')
    plt.xlabel('Əhəmiyyət dərəcəsi')
    plt.ylabel('Xüsusiyyət')
    plt.tight_layout()
    plt.savefig(os.path.join(output_dir, 'feature_importance.png'), dpi=300, bbox_inches='tight')
    plt.close()

    print("\nTop 10 ən əhəmiyyətli xüsusiyyət:")
    print(feature_importance.head(10))

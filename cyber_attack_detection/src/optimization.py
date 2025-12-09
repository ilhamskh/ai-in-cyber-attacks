import numpy as np
import pandas as pd
import time
import matplotlib.pyplot as plt
from sklearn.metrics import confusion_matrix
import os

def optimize_threshold(model, X_test, y_test, y_pred_proba, output_dir='results'):
    """
    Threshold optimizasiyasını həyata keçirir.
    """
    alpha = 0.5  # TPR əmsalı
    beta = 0.3   # FPR cəzası əmsalı
    gamma = 0.2  # Hesablama xərcləri əmsalı

    # Hesablama xərclərinin hesablanması (inference time)
    start_time = time.time()
    _ = model.predict(X_test[:1000])
    inference_time = time.time() - start_time
    avg_inference_time = inference_time / 1000  # hər nümunə üçün

    # Müxtəlif threshold dəyərləri üçün analiz
    thresholds_test = np.arange(0.3, 0.8, 0.05)
    results = []

    for threshold in thresholds_test:
        y_pred_threshold = (y_pred_proba >= threshold).astype(int)
        cm_temp = confusion_matrix(y_test, y_pred_threshold)
        tn, fp, fn, tp = cm_temp.ravel()
        
        tpr_temp = tp / (tp + fn)
        fpr_temp = fp / (fp + tn)
        precision_temp = tp / (tp + fp) if (tp + fp) > 0 else 0
        
        # Məqsəd funksiyası
        J_temp = alpha * tpr_temp - beta * fpr_temp - gamma * avg_inference_time
        
        results.append({
            'threshold': threshold,
            'TPR': tpr_temp,
            'FPR': fpr_temp,
            'Precision': precision_temp,
            'J(θ)': J_temp
        })

    results_df = pd.DataFrame(results)
    optimal_threshold = results_df.loc[results_df['J(θ)'].idxmax(), 'threshold']
    max_j_theta = results_df['J(θ)'].max()

    print(f"\nOptimal threshold: {optimal_threshold:.2f}")
    print("\nThreshold optimizasiya nəticələri:")
    print(results_df.to_string(index=False))

    # Vizualizasiya
    fig, ax1 = plt.subplots(figsize=(10, 6))
    ax1.plot(results_df['threshold'], results_df['TPR'], 
             'b-', label='TPR', linewidth=2)
    ax1.plot(results_df['threshold'], results_df['FPR'], 
             'r-', label='FPR', linewidth=2)
    ax1.set_xlabel('Threshold')
    ax1.set_ylabel('Dərəcə')
    ax1.legend(loc='upper left')

    ax2 = ax1.twinx()
    ax2.plot(results_df['threshold'], results_df['J(θ)'], 
             'g--', label='J(θ)', linewidth=2)
    ax2.set_ylabel('Məqsəd funksiyası', color='g')
    ax2.legend(loc='upper right')

    plt.title('Threshold Optimizasiyası')
    plt.savefig(os.path.join(output_dir, 'threshold_optimization.png'), dpi=300, bbox_inches='tight')
    plt.close()

    return optimal_threshold, max_j_theta, avg_inference_time

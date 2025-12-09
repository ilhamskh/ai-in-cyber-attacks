import time
import numpy as np
import pandas as pd

def detect_intrusion(network_traffic, model, preprocessor, threshold=0.5):
    """
    Real vaxt kiberhücum aşkarlama funksiyası
    """
    # Preprocessor class handles transformation
    # network_traffic is a dict
    
    # Normalizasiya və kodlaşdırma
    traffic_scaled = preprocessor.transform_single(network_traffic)
    
    # Proqnoz
    attack_proba = model.predict_proba(traffic_scaled)[0, 1]
    is_attack = attack_proba >= threshold
    
    # Risk səviyyəsinin müəyyən edilməsi
    if attack_proba < 0.3:
        risk_level = "AŞAĞI"
    elif attack_proba < 0.6:
        risk_level = "ORTA"
    else:
        risk_level = "YÜKSƏK"
    
    return {
        'is_attack': is_attack,
        'attack_probability': attack_proba,
        'risk_level': risk_level,
        'recommendation': "BLOKLANMALIDIR" if is_attack else "İCAZƏ VERİLİR"
    }

def run_simulation(model, preprocessor, X_test_original, y_test, optimal_threshold, n_samples=20):
    """
    Simulyasiyanı işə salır.
    """
    # Test nümunələri (raw data needed for simulation input simulation)
    # X_test_original should be the dataframe before scaling but after encoding?
    # Actually, the user code takes `X_test` which was `train_df.drop('label')`.
    # But `detect_intrusion` in user code does encoding inside.
    # So we need raw data.
    
    # In our main flow, we will pass the raw X_test (before scaling/encoding if possible, or handle it).
    # The user's `detect_intrusion` takes a dict and does encoding/scaling.
    # So we need to pass raw data dicts.
    
    # Let's assume X_test_original is the raw dataframe (before encoding/scaling).
    
    print("=== REAL VAXT SİMULYASİYA NƏTİCƏLƏRİ ===\n")
    
    # Take first 5 for detailed view
    test_samples = X_test_original.iloc[:5].to_dict('records')
    
    for i, sample in enumerate(test_samples, 1):
        result = detect_intrusion(sample, model, preprocessor, threshold=optimal_threshold)
        
        print(f"Nümunə #{i}:")
        # We need the true label corresponding to this sample
        true_label_val = y_test.iloc[i-1]
        print(f"  Həqiqi etiket: {'Hücum' if true_label_val==1 else 'Normal'}")
        print(f"  Proqnoz: {'HÜCUM' if result['is_attack'] else 'NORMAL'}")
        print(f"  Ehtimal: {result['attack_probability']:.4f}")
        print(f"  Risk səviyyəsi: {result['risk_level']}")
        print(f"  Tövsiyə: {result['recommendation']}")
        print("-" * 50)

    # Streaming simulation
    print("\n=== ŞƏBƏKƏ MONİTORİNQİ BAŞLADI ===\n")
    
    attack_count = 0
    normal_count = 0
    
    for i in range(n_samples):
        # Random test nümunəsi seç
        idx = np.random.randint(0, len(X_test_original))
        sample = X_test_original.iloc[idx].to_dict()
        true_label = y_test.iloc[idx]
        
        # Aşkarlama
        result = detect_intrusion(sample, model, preprocessor, threshold=optimal_threshold)
        
        if result['is_attack']:
            attack_count += 1
            status = "🚨 HÜCUM AŞKAR EDİLDİ"
        else:
            normal_count += 1
            status = "✓ Normal trafik"
        
        print(f"Paket #{i+1:03d} | {status} | "
              f"Ehtimal: {result['attack_probability']:.3f} | "
              f"Risk: {result['risk_level']}")
        
        # Qısa simulyasiya gecikmesi
        time.sleep(0.1)
    
    print(f"\n=== MONİTORİNQ XÜLASƏSI ===")
    print(f"Ümumi paketlər: {n_samples}")
    print(f"Normal: {normal_count} ({normal_count/n_samples*100:.1f}%)")
    print(f"Hücum: {attack_count} ({attack_count/n_samples*100:.1f}%)")

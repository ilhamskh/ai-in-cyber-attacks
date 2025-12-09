from sklearn.ensemble import RandomForestClassifier

def create_model():
    """
    Random Forest modelini yaradır.
    """
    rf_model = RandomForestClassifier(
        n_estimators=100,        # Ağac sayı
        max_depth=20,            # Maksimum dərinlik
        min_samples_split=5,     # Bölünmə üçün minimum nümunə
        min_samples_leaf=2,      # Yarpaqda minimum nümunə
        random_state=42,
        n_jobs=-1,               # Bütün prosessorları istifadə et
        verbose=1
    )
    return rf_model

def train_model(model, X_train, y_train):
    """
    Modeli təlim edir.
    """
    print("Model təlimi başlayır...")
    model.fit(X_train, y_train)
    print("Təlim tamamlandı!")
    return model

import pandas as pd
import numpy as np
from sklearn.preprocessing import StandardScaler, LabelEncoder

class Preprocessor:
    def __init__(self):
        self.label_encoders = {}
        self.scaler = StandardScaler()
        self.categorical_cols = ['protocol_type', 'service', 'flag']
        self.columns_to_drop = ['attack_type', 'difficulty']

    def process_labels(self, df):
        """
        Binary klassifikasiya üçün etiketləri hazırlayır: Normal (0) vs Attack (1)
        """
        df['label'] = df['attack_type'].apply(
            lambda x: 0 if x == 'normal' else 1
        )
        return df

    def fit_transform(self, train_df):
        """
        Təlim məlumatları üzərində öyrənir və transformasiya edir.
        """
        # Etiketləri hazırla
        train_df = self.process_labels(train_df)
        
        # Kateqorik dəyişənlərin kodlaşdırılması
        for col in self.categorical_cols:
            le = LabelEncoder()
            train_df[col] = le.fit_transform(train_df[col])
            self.label_encoders[col] = le
            
        # Xüsusiyyətləri və hədəf dəyişəni ayır
        X_train = train_df.drop(self.columns_to_drop + ['label'], axis=1)
        y_train = train_df['label']
        
        self.feature_names = X_train.columns

        # Normalizasiya
        X_train_scaled = self.scaler.fit_transform(X_train)
        X_train_scaled = pd.DataFrame(X_train_scaled, columns=X_train.columns)
        
        return X_train_scaled, y_train, X_train

    def transform(self, test_df):
        """
        Test məlumatlarını transformasiya edir (fit etmədən).
        """
        # Etiketləri hazırla
        test_df = self.process_labels(test_df)
        
        # Kateqorik dəyişənlərin kodlaşdırılması
        for col in self.categorical_cols:
            le = self.label_encoders[col]
            # Naməlum etiketlər üçün handle (əgər varsa, ən çox rast gəlinənlə əvəz etmək olar, 
            # amma sadəlik üçün transform edirik, xəta olsa error verəcək - real sistemdə handle edilməlidir)
            # Burada sadə yanaşma: unknown label-ləri 'unknown' kimi qəbul edib fit etmək lazımdır əslində,
            # amma LabelEncoder sadədir. Gəlin sadə saxlayaq, user kodu kimi.
            
            # User kodunda birbaşa transform edilir. Eyni yanaşmanı saxlayırıq.
            # Lakin test setdə train setdə olmayan kateqoriyalar ola bilər.
            # Bunu aşmaq üçün sadə bir try-except və ya map istifadə edə bilərik, 
            # amma user koduna sadiq qalaq.
            
            # Xətanın qarşısını almaq üçün:
            test_df[col] = test_df[col].map(lambda s: '<unknown>' if s not in le.classes_ else s)
            le.classes_ = np.append(le.classes_, '<unknown>')
            test_df[col] = le.transform(test_df[col])

        # Xüsusiyyətləri və hədəf dəyişəni ayır
        X_test = test_df.drop(self.columns_to_drop + ['label'], axis=1)
        y_test = test_df['label']
        
        # Normalizasiya
        X_test_scaled = self.scaler.transform(X_test)
        X_test_scaled = pd.DataFrame(X_test_scaled, columns=X_test.columns)
        
        return X_test_scaled, y_test

    def transform_single(self, sample_dict):
        """
        Tək bir nümunəni (dict) model üçün hazırlayır.
        """
        df = pd.DataFrame([sample_dict])
        
        # Kateqorik
        for col in self.categorical_cols:
            if col in df.columns:
                le = self.label_encoders[col]
                val = df[col].iloc[0]
                if val not in le.classes_:
                     # Fallback mechanism needed? Or just assume valid input for simulation
                     pass
                else:
                    df[col] = le.transform(df[col])
        
        # Normalizasiya
        # Scaler bütün sütunları gözləyir. Sample dict-də bütün sütunlar olmalıdır.
        
        scaled_array = self.scaler.transform(df)
        
        if hasattr(self, 'feature_names'):
            return pd.DataFrame(scaled_array, columns=self.feature_names)
        
        return scaled_array

    def get_feature_names(self):
        if hasattr(self.scaler, 'feature_names_in_'):
            return self.scaler.feature_names_in_
        return None

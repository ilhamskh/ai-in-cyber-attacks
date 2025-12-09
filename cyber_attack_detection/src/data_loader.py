import pandas as pd
import os

def load_data(data_dir='data'):
    """
    NSL-KDD məlumat dəstini yükləyir.
    """
    columns = ['duration', 'protocol_type', 'service', 'flag', 'src_bytes', 
               'dst_bytes', 'land', 'wrong_fragment', 'urgent', 'hot',
               'num_failed_logins', 'logged_in', 'num_compromised', 
               'root_shell', 'su_attempted', 'num_root', 'num_file_creations',
               'num_shells', 'num_access_files', 'num_outbound_cmds',
               'is_host_login', 'is_guest_login', 'count', 'srv_count',
               'serror_rate', 'srv_serror_rate', 'rerror_rate', 
               'srv_rerror_rate', 'same_srv_rate', 'diff_srv_rate',
               'srv_diff_host_rate', 'dst_host_count', 'dst_host_srv_count',
               'dst_host_same_srv_rate', 'dst_host_diff_srv_rate',
               'dst_host_same_src_port_rate', 'dst_host_srv_diff_host_rate',
               'dst_host_serror_rate', 'dst_host_srv_serror_rate',
               'dst_host_rerror_rate', 'dst_host_srv_rerror_rate',
               'attack_type', 'difficulty']

    train_path = os.path.join(data_dir, 'KDDTrain+.txt')
    test_path = os.path.join(data_dir, 'KDDTest+.txt')

    if not os.path.exists(train_path) or not os.path.exists(test_path):
        raise FileNotFoundError(f"Dataset faylları tapılmadı: {train_path} və ya {test_path}")

    print("Məlumatlar yüklənir...")
    train_df = pd.read_csv(train_path, names=columns, header=None)
    test_df = pd.read_csv(test_path, names=columns, header=None)

    print(f"Təlim dataseti ölçüsü: {train_df.shape}")
    print(f"Test dataseti ölçüsü: {test_df.shape}")

    return train_df, test_df

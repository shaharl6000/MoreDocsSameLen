import numpy as np
import os













if __name__ == "__main__":
    pass
    data = np.load("/cs/labs/tomhope/nirm/MusiQue/pkl_files_original_questions/datasets_orig_q_Expanded_Set.pkl", allow_pickle= True)
    print(data)
    file_path = "/cs/labs/tomhope/nirm/MusiQue/pkl_files_original_questions/datasets_orig_q_replaced_Hybrid8.pkl"
    file_size_bytes = os.path.getsize(file_path)
    file_size_mb = file_size_bytes / (1024 * 1024)

    print(f"Size of the local file: {file_size_mb:.2f} MB")
    # data = np.load("/cs/labs/tomhope/nirm/MusiQue/pkl_files/datasets_extendedSet.pkl", allow_pickle=True)
    # print(data)

    # data = np.load("/cs/labs/tomhope/nirm/MusiQue/pkl_files/datasets_fullSet.pkl", allow_pickle=True)
    # print(data)

    # data = np.load("/cs/labs/tomhope/nirm/MusiQue/pkl_files/datasets_noDocSet.pkl", allow_pickle=True)
    # print(data)

    # data = np.load("/cs/labs/tomhope/nirm/MusiQue/pkl_files/datasets_pad_token_mistral.pkl", allow_pickle=True)
    # print(data)

    # data = np.load("/cs/labs/tomhope/nirm/MusiQue/pkl_files/datasets_raplacedSet.pkl", allow_pickle=True)
    # print(data)

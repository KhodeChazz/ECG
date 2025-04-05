import os
import requests
from tqdm import tqdm

class ECGDatasetDownloader:
    def __init__(self, base_url="https://physionet.org/files/mitdb/1.0.0/", save_dir="mitdb"):
        self.base_url = base_url
        self.save_dir = save_dir
        self.record_numbers = [
            '100', '101', '102', '103', '104', '105', '106', '107', '108', '109',
            '111', '112', '113', '114', '115', '116', '117', '118', '119', '121',
            '122', '123', '124', '200', '201', '202', '203', '205', '207', '208',
            '209', '210', '212', '213', '214', '215', '217', '219', '220', '221',
            '222', '223', '228', '230', '231', '232', '233', '234'
        ]
        self.file_types = ['dat', 'hea', 'atr']
        
    def download_dataset(self):
        os.makedirs(self.save_dir, exist_ok=True)
        
        for record in tqdm(self.record_numbers, desc="Downloading records"):
            for file_type in self.file_types:
                self._download_file(record, file_type)
    
    def _download_file(self, record, file_type):
        file_name = f"{record}.{file_type}"
        file_url = self.base_url + file_name
        save_path = os.path.join(self.save_dir, file_name)

        if not os.path.exists(save_path):
            response = requests.get(file_url, stream=True)
            with open(save_path, 'wb') as f:
                f.write(response.content)
            print(f"Downloaded {file_name}")
        else:
            print(f"{file_name} already exists, skipping download.")
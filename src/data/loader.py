# src/data/loader.py

import os
import re
import pandas as pd
import numpy as np
from datetime import datetime, timedelta
from typing import Tuple, List, Optional
import warnings

class SatelliteDataLoader:
    """
  - Fixed loading issues with the TLE epoch column.
  - Included smart dispatchers to parse multiple different motor file formats.
    """
    
    def __init__(self, data_dir: str = "data"):
        self.data_dir = data_dir
        self.tle_dir = os.path.join(data_dir, "orbital_elements")
        self.maneuver_dir = os.path.join(data_dir, "manoeuvres")
        
        self.MANEUVER_FILE_MAP = {
            'cryosat-2': 'cs2man.txt', 'haiyang-2a': 'h2aman.txt', 'jason-1': 'ja1man.txt',
            'jason-2': 'ja2man.txt', 'jason-3': 'ja3man.txt', 'saral': 'srlman.txt',
            'sentinel-3a': 's3aman.txt', 'sentinel-3b': 's3bman.txt', 'spot-2':'sp2man.txt',
            'sentinel-6a': 's6aman.txt', 'topex': 'topman.txt', 'spot-4': 'sp4man.txt',
            'spot-5': 'sp5man.txt'
        }
        os.makedirs(os.path.join(data_dir, "processed"), exist_ok=True)

    def load_satellite_data(self, satellite_name: str) -> Tuple[pd.DataFrame, List[datetime]]:
        tle_data = self._load_tle_data(satellite_name)
        maneuver_data = self._load_maneuver_data(satellite_name)
        print(f"📊 Loaded {len(tle_data)} TLE records and {len(maneuver_data)} maneuvers for {satellite_name}")
        return tle_data, maneuver_data

    def _load_tle_data(self, satellite_name: str) -> pd.DataFrame:
        tle_filename = f"{satellite_name}.csv"
        raw_path = os.path.join(self.tle_dir, tle_filename)
        if not os.path.exists(raw_path):
            raise FileNotFoundError(f"No TLE file found for {satellite_name} at {raw_path}")
        
        tle_data = pd.read_csv(raw_path)
        
        column_mapping = {'Brouwer mean motion': 'mean_motion', 'argument of perigee': 'argument_of_perigee', 
                          'mean anomaly': 'mean_anomaly', 'right ascension': 'right_ascension'}
        tle_data = tle_data.rename(columns=column_mapping)
        
        numeric_cols = ['mean_motion', 'eccentricity', 'inclination', 'argument_of_perigee', 
                        'right_ascension', 'mean_anomaly', 'bstar']
        print("   -> Verifying and coercing data types to numeric...")
        for col in numeric_cols:
            if col in tle_data.columns:
                tle_data[col] = pd.to_numeric(tle_data[col], errors='coerce')
                if tle_data[col].isna().any():
                    warnings.warn(f"      - Column '{col}' has NaNs after coercion.")
        

        if 'epoch' not in tle_data.columns and not tle_data.empty:
            # If there is no 'epoch' column, the first column is assumed to be epoch
            first_col_name = tle_data.columns[0]
            tle_data = tle_data.rename(columns={first_col_name: 'epoch'})
            warnings.warn(f"      - 'epoch' column not found. Assuming first column ('{first_col_name}') is the epoch.")
        # ---------------------------------------------------

        if 'epoch' in tle_data.columns:
            tle_data['epoch'] = pd.to_datetime(tle_data['epoch'])
        else:
            # If there is still no epoch column after the backup logic is processed, an error is thrown
            raise ValueError("Epoch column not found.")

        return tle_data.sort_values('epoch').reset_index(drop=True)
    
    def _load_maneuver_data(self, satellite_name: str) -> List[datetime]:
        file_path = self._get_maneuver_file_path(satellite_name)
        if not file_path or not os.path.exists(file_path):
            warnings.warn(f"Maneuver file not found for {satellite_name}")
            return []
        
        print(f"✅ Found maneuver file. Loading from: {file_path}")
        return self._parse_maneuver_file(file_path)

    def _get_maneuver_file_path(self, satellite_name: str) -> Optional[str]:
        clean_name = satellite_name.lower().strip().replace('_', '-')
        filename = None
        
        if 'fengyun' in clean_name:
            fn_name = satellite_name.replace('_', '-')
            filename = f"man{fn_name}.txt"
        else:
            filename = self.MANEUVER_FILE_MAP.get(clean_name)
        
        return os.path.join(self.maneuver_dir, filename) if filename else None

    def _parse_fengyun_format(self, line: str) -> Optional[datetime]:
        match = re.search(r'(\d{4}-\d{2}-\d{2}T\d{2}:\d{2}:\d{2})', line)
        if match:
            try:
                return datetime.fromisoformat(match.group(1))
            except ValueError:
                return None
        return None

    def _parse_doy_format(self, line: str) -> Optional[datetime]:
        match = re.search(r'(\d{4})\s+(\d{1,3})\s+(\d{1,2})\s+(\d{1,2})', line)
        if match:
            try:
                year, doy, hour, minute = match.groups()
                date_str = f"{year} {doy} {hour} {minute}"
                return datetime.strptime(date_str, "%Y %j %H %M")
            except ValueError:
                return None
        return None

    def _parse_maneuver_file(self, file_path: str) -> List[datetime]:
        maneuver_times = []
        filename = os.path.basename(file_path).lower()

        if 'fengyun' in filename:
            parser = self._parse_fengyun_format
            print("   -> Using Fengyun (ISO format) parser.")
        else:
            parser = self._parse_doy_format
            print("   -> Using Day-of-Year (DOY format) parser.")
            
        try:
            with open(file_path, 'r', encoding='utf-8', errors='ignore') as f:
                for line in f:
                    line = line.strip()
                    if not line or line.startswith('#'):
                        continue
                    
                    dt = parser(line)
                    if dt:
                        maneuver_times.append(dt)
        except Exception as e:
            warnings.warn(f"Error reading or parsing maneuver file {file_path}: {e}")
        
        unique_maneuvers = sorted(list(set(maneuver_times)))
        print(f"   -> Successfully parsed {len(unique_maneuvers)} unique maneuvers from the file.")
        return unique_maneuvers
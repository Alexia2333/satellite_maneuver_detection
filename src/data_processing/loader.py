# src/data/loader.py

import os
import re
import pandas as pd
import numpy as np
from datetime import datetime, timedelta
from typing import Tuple, List, Optional
from sgp4.api import Satrec
from sgp4.conveniences import sat_epoch_datetime
import warnings

class SatelliteDataLoader:
    """
    Loads satellite TLE and maneuver data.
    (Final universal version capable of parsing all provided maneuver file formats)
    """
    
    def __init__(self, data_dir: str = "data"):
        self.data_dir = data_dir
        self.tle_dir = os.path.join(data_dir, "raw", "TLE")
        self.processed_dir = os.path.join(data_dir, "processed")
        self.maneuver_dir = os.path.join(data_dir, "manoeuvres")
        
        self.MANEUVER_FILE_MAP = {
            'cryosat-2': 'cs2man.txt', 'haiyang-2a': 'h2aman.txt', 'jason-1': 'ja1man.txt',
            'jason-2': 'ja2man.txt', 'jason-3': 'ja3man.txt', 'saral': 'srlman.txt',
            'sentinel-3a': 's3aman.txt', 'sentinel-3b': 's3bman.txt', 'spot-2':'sp2man.txt',
            'sentinel-6a': 's6aman.txt', 'topex': 'topman.txt', 'spot-4': 'sp4man.txt',
            'spot-5': 'sp5man.txt'
        }
        os.makedirs(self.processed_dir, exist_ok=True)

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
        
        # 1. 首先重命名列，以便使用标准化的列名
        column_mapping = self._get_column_mapping()
        tle_data = tle_data.rename(columns=column_mapping)
        
        # 2. 【核心修复】强制将所有轨道参数列转换为数值类型
        #    定义所有应该为数字的列
        numeric_cols = [
            'mean_motion', 'eccentricity', 'inclination', 'argument_of_perigee', 
            'right_ascension', 'mean_anomaly', 'bstar'
        ]
        
        print("   -> Verifying and coercing data types to numeric...")
        for col in numeric_cols:
            if col in tle_data.columns:
                # 使用 pd.to_numeric 进行转换，任何无法转换的值 (errors) 都会被强制 (coerce) 变成 NaN
                tle_data[col] = pd.to_numeric(tle_data[col], errors='coerce')
                
                # 报告此操作可能产生的NaN数量
                nan_count = tle_data[col].isna().sum()
                if nan_count > 0:
                    warnings.warn(f"      - Column '{col}' has {nan_count} NaN values after coercion. These will be handled by interpolation.")
            else:
                 # 如果关键列不存在，也打印一个警告
                 if col in ['mean_motion', 'eccentricity', 'inclination']:
                     warnings.warn(f"      - Critical column '{col}' not found in the TLE data.")

        # 3. 处理epoch列
        if 'epoch' not in tle_data.columns and not tle_data.empty:
            tle_data = tle_data.rename(columns={tle_data.columns[0]: 'epoch'})
        
        if 'epoch' in tle_data.columns:
            tle_data['epoch'] = pd.to_datetime(tle_data['epoch'])
        else:
            raise ValueError("Epoch column could not be found or created.")

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
            # Construct filename for Fengyun, e.g., "manFengyun-4A.txt"
            fn_name = satellite_name.replace('_', '-')
            filename = f"man{fn_name}.txt"
        else:
            filename = self.MANEUVER_FILE_MAP.get(clean_name)
        
        if filename:
            return os.path.join(self.maneuver_dir, filename)
        return None

    def _get_column_mapping(self) -> dict:
        return {'Brouwer mean motion': 'mean_motion', 'argument of perigee': 'argument_of_perigee', 
                'mean anomaly': 'mean_anomaly', 'right ascension': 'right_ascension'}

    def _parse_datetime_string(self, dt_str: str) -> Optional[datetime]:
        """Tries to parse a datetime string with several common formats."""
        # Clean up the string to extract a potential datetime pattern
        # This regex looks for patterns like 'YYYY-MM-DD HH:MM:SS'
        match = re.search(r'(\d{4}[-/]\d{2}[-/]\d{2}[ T]\d{2}:\d{2}:\d{2})', dt_str)
        if match:
            dt_str = match.group(1).replace('/', '-')
        
        formats_to_try = [
            '%Y-%m-%d %H:%M:%S',
            '%Y-%m-%dT%H:%M:%S',
        ]
        
        for fmt in formats_to_try:
            try:
                # Handle potential fractional seconds
                timestamp = dt_str.split('.')[0]
                return datetime.strptime(timestamp, fmt)
            except ValueError:
                continue
        return None
        
    def _parse_maneuver_file(self, file_path: str) -> List[datetime]:
        """
        Parses a maneuver file to extract maneuver times.
        This universal version can handle multiple formats, including:
        - Standard format (YYYY-MM-DD HH:MM:SS) used by Fengyun series.
        - Special Jason series format (YYYY DDD HH MM).
        """
        maneuver_times = []
        filename = os.path.basename(file_path) # Get just the filename

        try:
            with open(file_path, 'r') as f:
                for line in f:
                    line = line.strip()
                    if not line or line.startswith('#'):
                        continue
                    
                    dt = None
                    # --- NEW Universal Logic ---

                    # 1. Check if it's a Jason-series file (ja1man.txt, ja2man.txt, etc.)
                    #    and the line starts with the correct satellite identifier.
                    if filename.startswith('ja') and filename.endswith('man.txt'):
                        if not line.upper().startswith('JASO'):
                            continue # Skip detail lines that don't start with JASO
                        try:
                            parts = line.split()
                            date_str_to_parse = f"{parts[1]} {parts[2]} {parts[3]} {parts[4]}"
                            dt = datetime.strptime(date_str_to_parse, "%Y %j %H %M")
                        except (IndexError, ValueError):
                            pass
                    
                    # 2. If it's not a Jason file or parsing failed, try the general-purpose parser.
                    if dt is None:
                        dt = self._parse_datetime_string(line)

                    # 3. Add the successfully parsed datetime to our list
                    if dt:
                        maneuver_times.append(dt)
                    else:
                        warnings.warn(f"Could not parse date from line: '{line}' in {file_path}")

        except Exception as e:
            warnings.warn(f"Error reading or parsing maneuver file {file_path}: {e}")
        
        maneuver_count = len(set(maneuver_times))
        print(f"   -> Successfully parsed {maneuver_count} unique maneuvers from the file.")
        return sorted(list(set(maneuver_times)))

    def _get_satellite_name_from_path(self, file_path: str) -> str:
        """Helper to get satellite name from TLE filename."""
        base_name = os.path.basename(file_path)
        return os.path.splitext(base_name)[0]

    def _parse_tle_file(self, file_path: str) -> Optional[pd.DataFrame]:
        """
        Parses a raw TLE file using the sgp4 library.
        NOTE: This method is not currently used by load_satellite_data, which loads a pre-made CSV.
        """
        satellite_name = self._get_satellite_name_from_path(file_path)
        records = []

        try:
            with open(file_path, 'r') as f:
                lines = f.readlines()

            for i in range(0, len(lines), 2):
                if i + 1 < len(lines):
                    line1 = lines[i].strip()
                    line2 = lines[i+1].strip()

                    if line1.startswith('1 ') and line2.startswith('2 '):
                        try:
                            sat = Satrec.from_twoline(line2, line1)
                            record = {
                                'satellite': satellite_name,
                                'epoch': sat_epoch_datetime(sat.epoch),
                                'mean_motion': sat.no_kozai,
                                'eccentricity': sat.ecco,
                                'inclination': np.rad2deg(sat.inclo),
                                'arg_perigee': np.rad2deg(sat.argpo),
                                'raan': np.rad2deg(sat.nodeo),
                                'mean_anomaly': np.rad2deg(sat.mo),
                                'bstar': sat.bstar
                            }
                            records.append(record)
                        except Exception as e:
                            warnings.warn(f"Skipping invalid TLE entry in {file_path}: {e}")
            
            return pd.DataFrame(records) if records else None

        except Exception as e:
            warnings.warn(f"Could not process file {file_path}: {e}")
            return None
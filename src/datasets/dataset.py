import h5py
import torch
from torch import nn
import torch.nn.functional as F
from torch.utils.data import Dataset

from torchvision import transforms as T, utils
from tqdm import tqdm

import pandas as pd

from src.utils.data_utils import get_neighboring_slices, multislice_data_minimal_process, \
                                                    get_3d_image_transform, crop_minimally_keep_ratio, rescale_intensity, rescale_intensity_3D, \
                                                    resample_and_reshape, process_tabular_data, data_minimal_process, CropOrPad_3D

import nibabel as nib
import numpy as np
import pandas as pd
import os

import logging

LOG = logging.getLogger(__name__)

# constants

DIAGNOSIS_MAP = {"CN": 0, "Dementia": 1, "AD": 1, "MCI": 2}
DIAGNOSIS_MAP_binary = {"CN": 0, "AD": 1, "Dementia": 1}

# Helper function to load numpy data from path
def load_numpy_from_path(path, data_type='3d'):
    """
    Load numpy data from file path
    Args:
        path: string path to numpy file
        data_type: '3d' for MRI/PET, '4d' for fMRI, '2d' for FC
    """
    try:
        if isinstance(path, bytes):
            path = path.decode('utf-8')
        
        data = np.load(path)
        
        # 4D 데이터를 3D로 올바르게 변환
        if data_type == '3d' and data.ndim == 4:
            # (1, 96, 112, 96) -> (96, 112, 96)로 변환
            if data.shape[0] == 1:
                data = data.squeeze(0)
            elif data.shape[1] == 1:
                # (15, 1, 96, 96) -> (96, 96, 15)로 변환하지 말고
                # 올바른 순서로 reshape
                data = data.squeeze(1)  # (15, 96, 96)
                # 축 순서를 바꿔서 (96, 96, 15) -> (96, 96, 15) 형태로 만들어야 함
                # 하지만 이는 데이터에 따라 다를 수 있으므로 주의 필요
            LOG.warning(f"Converted 4D data to 3D for path: {path}, new shape: {data.shape}")
        
        # Validate data dimensions based on type
        if data_type == '3d' and data.ndim != 3:
            LOG.warning(f"Expected 3D data but got {data.ndim}D for path: {path}")
        elif data_type == '4d' and data.ndim != 4:
            LOG.warning(f"Expected 4D data but got {data.ndim}D for path: {path}")
        elif data_type == '2d' and data.ndim != 2:
            LOG.warning(f"Expected 2D data but got {data.ndim}D for path: {path}")
            
        return data
    except Exception as e:
        LOG.error(f"Failed to load numpy data from {path}: {str(e)}")
        return None
# dataset
# PET_shape = (self.resolution[0], self.resolution[1], self.resolution[0])
# MRI_shape = (self.resolution[0], self.resolution[1], self.resolution[0])

class SlicedScanMRI2PETDataset(Dataset):
    def __init__(
        self,
        resolution = None,
        mri_root_path = '/path/to/mri',
        pet_root_path = '/path/to/pet',
        data_path = None,
        output_dim = 1,
        direction = 'coronal',
        standardized_tabular = True,
        classes=None, # 'binary' or 'multi' (with MCI)
        random_crop=False,
        random_flip=False,
        random_affine=False,
        resample_mri=False,
        dx_labels = ['CN', 'MCI', 'Dementia'],
        use_fmri = False,
        use_fc = False,
    ):
        super().__init__()
        self.resolution = resolution
        self.mri_root_path = mri_root_path
        self.pet_root_path = pet_root_path
        self.data_path = data_path
        self.output_dim = output_dim
        self.direction = direction
        self.standardized_tabular = standardized_tabular
        self.random_crop = random_crop
        self.random_flip = random_flip
        self.random_affine = random_affine
        self.with_label = classes
        self.resample_mri = resample_mri
        self.dx_labels = dx_labels
        self.use_fmri = use_fmri
        self.use_fc = use_fc

        self._load()

    def _load(self):
        mri_data = []
        pet_data = []
        diagnosis = []
        tabular_data = []
        mri_uid = []
        pet_uid = []  # PET 파일명을 저장할 리스트 추가
        
        # New data lists for fMRI and FC
        fmri_data = []
        fc_data = []

        PET_shape = (self.resolution[0], self.resolution[1], self.resolution[0])
        MRI_shape = (self.resolution[0], self.resolution[1], self.resolution[0])

        if self.data_path is not None and 'h5' in self.data_path:       
            print('loaded from h5 file with path-based loading')
            with h5py.File(self.data_path, mode='r') as file:
                for name, group in tqdm(file.items(), total=len(file)):
                    if name == "stats":
                        self.tabular_mean = group["tabular/mean"][:]
                        self.tabular_std = group["tabular/stddev"][:]
                    else:
                        if group.attrs['DX'] not in self.dx_labels:
                            continue
                        
                        # Load MRI data from path
                        mri_path = group['MRI/T1/path'][()]
                        input_mri_data = load_numpy_from_path(mri_path, '3d')
                        if input_mri_data is None:
                            continue
                            
                        if self.resample_mri:
                            _resampled_mri_data = resample_and_reshape(input_mri_data, (1.5, 1.5, 1.5), PET_shape)
                            input_mri_data = _resampled_mri_data
                            MRI_shape = PET_shape
                            assert input_mri_data.shape == PET_shape
                        
                        # Load PET data from path
                        pet_path = group['PET/FDG/path'][()]
                        pet_path_str = group['PET/FDG/path'][()].decode('utf-8')
                        
                        # PET 경로에서 파일명 추출
                        pet_filename = os.path.basename(pet_path_str)
                        
                        _pet_data = load_numpy_from_path(pet_path, '3d')
                        if _pet_data is None:
                            continue
                            
                        _mri_data = input_mri_data
                        _tabular_data = group['tabular'][:]
                        _diagnosis = group.attrs['DX']

                        _pet_data = np.nan_to_num(_pet_data, copy=False)
                        mri_data.append(_mri_data)
                        pet_data.append(_pet_data)
                        tabular_data.append(_tabular_data)
                        diagnosis.append(_diagnosis)
                        mri_uid.append(name)
                        pet_uid.append(pet_filename)  # PET 파일명 추가
                        
                        # Load fMRI data if enabled
                        if self.use_fmri and 'fMRI/BOLD/path' in group:
                            fmri_path = group['fMRI/BOLD/path'][()]
                            _fmri_data = load_numpy_from_path(fmri_path, '4d')
                            if _fmri_data is not None:
                                fmri_data.append(_fmri_data)
                            else:
                                fmri_data.append(np.zeros((91, 109, 91, 200)))  # Default shape for fMRI
                        
                        # Load FC data if enabled
                        if self.use_fc and 'FC/path' in group:
                            fc_path = group['FC/path'][()]
                            _fc_data = load_numpy_from_path(fc_path, '2d')
                            if _fc_data is not None:
                                assert _fc_data.shape == (300, 300), f"FC data should be 300x300, got {_fc_data.shape}"
                                fc_data.append(_fc_data)
                            else:
                                fc_data.append(np.zeros((300, 300)))  # Default FC matrix
        else:
            print('loaded from: ', self.mri_root_path, self.pet_root_path)

            mri_id = os.listdir(self.mri_root_path)
            mri_input = [os.path.join(self.mri_root_path, i, 'mri.nii.gz') for i in mri_id]
            pet_input = [os.path.join(self.pet_root_path, i[:-8], f'pet_fdg.nii.gz') for i in mri_id]
            mri_data = [nib.load(i).get_fdata() for i in mri_input]
            pet_data = [nib.load(i).get_fdata() for i in pet_input]

            csv_info = pd.read_csv('data_info.csv')
            diagnosis = [csv_info.loc[csv_info["IMAGEUID"] == i]['DX'].values[0] for i in mri_id]
            tabular_data = [csv_info.loc[csv_info["IMAGEUID"] == i]['TAB'].values[0] for i in mri_id]
            mri_uid = mri_id
            # PET 파일명도 생성 (MRI 기반)
            pet_uid = [uid.replace('_MRI', '_PET') if '_MRI' in uid else uid + '_PET' for uid in mri_id]

        self.len_data = len(pet_data)
        self._image_data_mri = mri_data
        self._image_data_pet = pet_data
        self._tabular_data = tabular_data
        self._mri_uid = mri_uid
        self._pet_uid = pet_uid  # PET 파일명 저장
        
        # Store additional data
        if self.use_fmri:
            self._fmri_data = fmri_data
        if self.use_fc:
            self._fc_data = fc_data
        
        LOG.info("DATASET: %s", self.data_path if self.data_path is not None else self.mri_root_path)
        LOG.info("SAMPLES: %d", self.len_data)
        if self.use_fmri:
            LOG.info("fMRI data loaded: %d samples", len(fmri_data))
        if self.use_fc:
            LOG.info("FC data loaded: %d samples", len(fc_data))

        # if self.with_label is not None:
        labels, counts = np.unique(diagnosis, return_counts=True)
        LOG.info("Classes: %s", pd.Series(counts, index=labels))     

        if self.with_label == 'binary':
            self._diagnosis = [DIAGNOSIS_MAP_binary[d] for d in diagnosis]
        elif self.with_label == 'multi':
            self._diagnosis = [DIAGNOSIS_MAP[d] for d in diagnosis]
        else:
            self._diagnosis = [DIAGNOSIS_MAP[d] for d in diagnosis]

    
    def __len__(self):
        return self.len_data


    def __getitem__(self, idx):

        MRI_shape = self.resolution
        PET_shape = self.resolution

        mri_scan = self._image_data_mri[idx]
        pet_scan = self._image_data_pet[idx]
        tabular_data = self._tabular_data[idx]
        mri_uid = self._mri_uid[idx]
        pet_uid = self._pet_uid[idx]  # PET 파일명 가져오기

        mri_scan = rescale_intensity_3D(mri_scan)
        pet_scan = rescale_intensity_3D(pet_scan)
        mri_scan = CropOrPad_3D(mri_scan, MRI_shape)
        pet_scan = CropOrPad_3D(pet_scan, PET_shape)

        if self.standardized_tabular:
            tabular_data = (tabular_data - self.tabular_mean) / self.tabular_std
            tabular_data = process_tabular_data(tabular_data)

        mri_scan_list = []
        pet_scan_list = []

        data_transform = T.Compose([
                T.ToTensor(),
                T.RandomVerticalFlip() if self.random_flip else nn.Identity(),
                T.RandomAffine(180, translate=(0.3, 0.3)) if self.random_affine else nn.Identity(),         
            ])

        if self.direction == 'coronal':
            for i in range(MRI_shape[1]):
                if self.output_dim > 1:
                    _mri_data = get_neighboring_slices(self.output_dim, self.direction, i, mri_scan)
                    _pet_data = get_neighboring_slices(self.output_dim, self.direction, i, pet_scan)
                    _pet_data = np.nan_to_num(_pet_data, copy=False)
                    _mri_data = multislice_data_minimal_process(self.output_dim, self.resolution, _mri_data, data_transform)
                    _pet_data = multislice_data_minimal_process(self.output_dim, self.resolution, _pet_data, data_transform)

                else:
                    _pet_data = pet_scan[:, i, :]
                    _mri_data = mri_scan[:, i, :]
                    _pet_data = np.nan_to_num(_pet_data, copy=False)            
                    _mri_data = data_minimal_process(self.resolution, _mri_data, data_transform)
                    _pet_data = data_minimal_process(self.resolution, _pet_data, data_transform)

                mri_scan_list.append(_mri_data)
                pet_scan_list.append(_pet_data)
        
        elif self.direction == 'sagittal':
            for i in range(MRI_shape[0]):
                if self.output_dim > 1:
                    _mri_data = get_neighboring_slices(self.output_dim, self.direction, i, mri_scan)
                    _pet_data = get_neighboring_slices(self.output_dim, self.direction, i, pet_scan)
                    _pet_data = np.nan_to_num(_pet_data, copy=False)
                    _mri_data = multislice_data_minimal_process(self.output_dim, self.resolution, _mri_data, data_transform)
                    _pet_data = multislice_data_minimal_process(self.output_dim, self.resolution, _pet_data, data_transform)
                else:
                    _pet_data = pet_scan[i, :, :]
                    _mri_data = mri_scan[i, :, :]
                    _pet_data = np.nan_to_num(_pet_data, copy=False)
                    _mri_data = data_minimal_process(self.resolution, _mri_data, data_transform)
                    _pet_data = data_minimal_process(self.resolution, _pet_data, data_transform)

                mri_scan_list.append(_mri_data)
                pet_scan_list.append(_pet_data)

        
        elif self.direction == 'axial':
            for i in range(MRI_shape[2]):
                if self.output_dim > 1:
                    _mri_data = get_neighboring_slices(self.output_dim, self.direction, i, mri_scan)
                    _pet_data = get_neighboring_slices(self.output_dim, self.direction, i, pet_scan)
                    
                    _pet_data = np.nan_to_num(_pet_data, copy=False)
                    _mri_data = multislice_data_minimal_process(self.output_dim, self.resolution, _mri_data, data_transform)
                    _pet_data = multislice_data_minimal_process(self.output_dim, self.resolution, _pet_data, data_transform)
                else:
                    _pet_data = pet_scan[:, :, i]
                    _mri_data = mri_scan[:, :, i]

                    _pet_data = np.nan_to_num(_pet_data, copy=False)
                    _mri_data = data_minimal_process(self.resolution, _mri_data, data_transform)
                    _pet_data = data_minimal_process(self.resolution, _pet_data, data_transform)

                mri_scan_list.append(_mri_data)
                pet_scan_list.append(_pet_data)

        label = self._diagnosis[idx]
        
        # Prepare return data
        return_data = [mri_scan_list, pet_scan_list, label, tabular_data, mri_uid, pet_uid]
        
        # Add fMRI data if enabled
        if self.use_fmri:
            fmri_data = self._fmri_data[idx]
            return_data.append(fmri_data)
        
        # Add FC data if enabled
        if self.use_fc:
            fc_data = self._fc_data[idx]
            return_data.append(fc_data)
       
        return tuple(return_data)

class MRI2PET_2_5D_Dataset(Dataset):
    def __init__(
        self,
        resolution = None,
        mri_root_path = '/path/to/mri',
        pet_root_path = '/path/to/pet',
        data_path = None,
        output_dim = 3,
        direction = 'axial',
        num_slices = 'all',
        standardized_tabular = True,
        classes=None, # 'binary' or 'multi' (with MCI)
        random_crop=False,
        random_flip=False,
        random_affine=False,
        resample_mri=False,
        ROI_mask = None,
        dx_labels = ['CN', 'Dementia', 'MCI'],
    ):
        super().__init__()
        self.resolution = resolution
        self.mri_root_path = mri_root_path
        self.pet_root_path = pet_root_path
        self.data_path = data_path
        self.output_dim = output_dim
        self.direction = direction
        self.num_slices = num_slices
        self.standardized_tabular = standardized_tabular
        self.random_crop = random_crop
        self.random_flip = random_flip
        self.random_affine = random_affine
        self.with_label = classes
        self.resample_mri = resample_mri

        self.ROI_mask = ROI_mask
        self.dx_labels = dx_labels

        self._load()

    def _load(self):
        mri_data = []
        pet_data = []
        diagnosis = []
        tabular_data = []
        slice_index = []
        mri_uid = []
        pet_uid = []  # PET 파일명을 저장할 리스트 추가

        PET_shape = (self.resolution[0], self.resolution[1], self.resolution[0])
        MRI_shape = (self.resolution[0], self.resolution[1], self.resolution[0])

        flag = 0

        if 'h5' in self.data_path:       
            print('loaded from h5 file')

            with h5py.File(self.data_path, mode='r') as file:
                for name, group in tqdm(file.items(), total=len(file)):

                    if name == "stats":
                        self.tabular_mean = group["tabular/mean"][:]
                        self.tabular_std = group["tabular/stddev"][:]
                    else:
                        if group.attrs['DX'] not in self.dx_labels:
                            continue
                            
                        # PET 경로에서 파일명 추출
                        if 'PET/FDG/path' in group:
                            pet_path_str = group['PET/FDG/path'][()].decode('utf-8')
                            pet_filename = os.path.basename(pet_path_str)
                        else:
                            pet_filename = name + '_PET'  # fallback
                            
                        if self.resample_mri:
                            _raw_mri_data = group['MRI/T1/data'][:]
                            _resampled_mri_data = resample_and_reshape(_raw_mri_data, (1.5, 1.5, 1.5), PET_shape)
                            input_mri_data = _resampled_mri_data
                            MRI_shape = PET_shape
                            assert input_mri_data.shape == PET_shape
                            input_pet_data = group['PET/FDG/data'][:]

                        else:
                            input_mri_data = group['MRI/T1/data'][:]
                            input_pet_data = group['PET/FDG/data'][:]

                        # 나머지 로직은 동일하지만 pet_uid도 함께 추가
                        if self.direction == 'coronal':
                            max_slice_index = PET_shape[1] - 1
                            if self.num_slices == 1:
                                _mri_data = get_neighboring_slices(self.output_dim, self.direction, PET_shape[1] // 2 + 1, input_mri_data)
                                _pet_data = get_neighboring_slices(self.output_dim, self.direction, PET_shape[1] // 2 + 1, input_pet_data)
                                _tabular_data = group['tabular'][:]
                                _diagnosis = group.attrs['DX']

                                _pet_data = np.nan_to_num(_pet_data, copy=False)
                                mri_data.append(_mri_data)
                                pet_data.append(_pet_data)
                                tabular_data.append(_tabular_data)
                                diagnosis.append(_diagnosis)
                                mri_uid.append(name)
                                pet_uid.append(pet_filename)  # PET 파일명 추가
                                slice_index.append(PET_shape[1] // 2 + 1)

                            elif self.num_slices == 'all':
                                for i in range(PET_shape[1]):
                                    _mri_data = get_neighboring_slices(self.output_dim, self.direction, i, input_mri_data)
                                    _pet_data = get_neighboring_slices(self.output_dim, self.direction, i, input_pet_data)
                                    
                                    _tabular_data = group['tabular'][:]
                                    _diagnosis = group.attrs['DX']

                                    _pet_data = np.nan_to_num(_pet_data, copy=False)
                                    if not np.any(_pet_data) or not np.any(_mri_data):
                                        continue
                                    mri_data.append(_mri_data)
                                    pet_data.append(_pet_data)
                                    tabular_data.append(_tabular_data)
                                    diagnosis.append(_diagnosis)
                                    mri_uid.append(name)
                                    pet_uid.append(pet_filename)  # PET 파일명 추가
                                    slice_index.append(i)
                            else:
                                for i in range(-self.num_slices // 2, self.num_slices // 2):
                                    _mri_data = get_neighboring_slices(self.output_dim, self.direction, PET_shape[1] // 2 + 1 + i, input_mri_data)
                                    _pet_data = get_neighboring_slices(self.output_dim, self.direction, PET_shape[1] // 2 + 1 + i, input_pet_data)
                                    _tabular_data = group['tabular'][:]
                                    _diagnosis = group.attrs['DX']

                                    _pet_data = np.nan_to_num(_pet_data, copy=False)
                                    if not np.any(_pet_data) or not np.any(_mri_data):
                                        continue

                                    mri_data.append(_mri_data)
                                    pet_data.append(_pet_data)
                                    tabular_data.append(_tabular_data)
                                    diagnosis.append(_diagnosis)
                                    mri_uid.append(name)
                                    pet_uid.append(pet_filename)  # PET 파일명 추가
                                    slice_index.append(PET_shape[1] // 2 + 1 + i)
                        
                        # sagittal과 axial 방향도 동일하게 pet_uid.append(pet_filename) 추가
                        elif self.direction == 'sagittal':
                            max_slice_index = PET_shape[0] - 1
                            # ... (동일한 로직에 pet_uid.append(pet_filename) 추가)
                        
                        elif self.direction == 'axial':
                            max_slice_index = PET_shape[2] - 1
                            # ... (동일한 로직에 pet_uid.append(pet_filename) 추가)

        else:
            raise NotImplementedError

        self.len_data = len(pet_data)
        self._image_data_mri = mri_data
        self._image_data_pet = pet_data
        self._tabular_data = tabular_data
        
        self._slice_index = [float(i / max_slice_index) for i in slice_index]
        self._max_slice_index = max_slice_index
        self._mri_uid = mri_uid
        self._pet_uid = pet_uid  # PET 파일명 저장
        
        # 나머지 코드는 동일...

    def __getitem__(self, idx):
        # ... 기존 코드 ...
        
        mri_scan = self._image_data_mri[idx]
        pet_scan = self._image_data_pet[idx]
        tabular_data = self._tabular_data[idx]
        
        slice_index = self._slice_index[idx]
        
        # ... 기존 처리 코드 ...

        label = self._diagnosis[idx]
        mri_uid = self._mri_uid[idx]
        pet_uid = self._pet_uid[idx]  # PET 파일명 가져오기

        return mri_scan, pet_scan, label, tabular_data, slice_index, loss_weight_mask, mri_uid, pet_uid

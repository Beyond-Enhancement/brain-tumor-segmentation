#!/usr/bin/env python3
"""Synthetic DICOM study from the demo NIfTI volumes, for the DICOM path tests.

Each sequence uses a different transfer syntax (uncompressed, JPEG 2000 lossless, RLE,
JPEG-LS lossless), plus a 3-slice localiser that the workbench should ignore. Patient tags hold
canary values so tests can check they never leave the page or reach an output file.
Usage: make_dicom.py <demo dir with t1/t1c/t2/flair.nii.gz> <out dir> [--uncompressed]"""
import os, sys, numpy as np, nibabel as nib, pydicom
from pydicom.dataset import Dataset, FileMetaDataset
from pydicom.uid import generate_uid, ExplicitVRLittleEndian, JPEG2000Lossless, RLELossless, JPEGLSLossless, MRImageStorage

SRC, OUT = sys.argv[1], sys.argv[2]
plain = '--uncompressed' in sys.argv
CANARY_NAME, CANARY_ID = 'CANARY^ZEBRA', 'CANARY-7741'
study = generate_uid()
SEQ = [  # file, description, TR, TE, TI, contrast, transfer syntax
    ('t1', 't1_se_tra', 550, 11, None, False, ExplicitVRLittleEndian),
    ('t1c', 't1_se_tra_post_gd', 550, 11, None, True, JPEG2000Lossless),
    ('t2', 't2_tse_tra', 4800, 98, None, False, RLELossless),
    ('flair', 't2_tirm_tra_dark-fluid', 9000, 85, 2500, False, JPEGLSLossless),
]


def write_series(arr, affine, desc, num, tr, te, ti, contrast, ts, folder):
    # arr: (i, j, k) voxel order of the NIfTI; DICOM rows = j, columns = i
    os.makedirs(folder, exist_ok=True)
    series = generate_uid(); frame = generate_uid()
    A = affine.copy(); A[:2, :] *= -1  # RAS -> LPS
    col_dir = A[:3, 0] / np.linalg.norm(A[:3, 0]); row_dir = A[:3, 1] / np.linalg.norm(A[:3, 1])
    dx, dy = np.linalg.norm(A[:3, 0]), np.linalg.norm(A[:3, 1]); dz = np.linalg.norm(A[:3, 2])
    a = np.clip(np.round(arr), 0, 65535).astype(np.uint16)
    order = np.random.default_rng(num).permutation(a.shape[2])  # files written in shuffled order
    for k in order:
        ds = Dataset()
        ds.file_meta = FileMetaDataset(); ds.file_meta.MediaStorageSOPClassUID = MRImageStorage
        ds.file_meta.MediaStorageSOPInstanceUID = generate_uid(); ds.file_meta.TransferSyntaxUID = ExplicitVRLittleEndian
        ds.SOPClassUID = MRImageStorage; ds.SOPInstanceUID = ds.file_meta.MediaStorageSOPInstanceUID
        ds.PatientName = CANARY_NAME; ds.PatientID = CANARY_ID; ds.PatientBirthDate = '19010101'; ds.PatientSex = 'O'
        ds.StudyInstanceUID = study; ds.SeriesInstanceUID = series; ds.FrameOfReferenceUID = frame
        ds.StudyDate = '20200101'; ds.Modality = 'MR'; ds.Manufacturer = 'SIEMENS'; ds.InstitutionName = 'CANARY HOSPITAL'
        ds.SeriesDescription = desc; ds.ProtocolName = desc; ds.SeriesNumber = num; ds.InstanceNumber = int(k) + 1
        ds.ImageType = ['ORIGINAL', 'PRIMARY', 'M', 'ND']; ds.MRAcquisitionType = '2D'; ds.ScanningSequence = 'SE'
        ds.RepetitionTime = tr; ds.EchoTime = te
        if ti: ds.InversionTime = ti
        if contrast: ds.ContrastBolusAgent = 'GADOVIST'
        ds.SliceThickness = float(dz); ds.MagneticFieldStrength = 3
        p = A[:3, :3] @ np.array([0, 0, k]) + A[:3, 3]
        ds.ImagePositionPatient = [float(v) for v in p]
        ds.ImageOrientationPatient = [float(v) for v in list(col_dir) + list(row_dir)]
        ds.PixelSpacing = [float(dy), float(dx)]
        ds.Rows, ds.Columns = a.shape[1], a.shape[0]
        ds.SamplesPerPixel = 1; ds.PhotometricInterpretation = 'MONOCHROME2'
        ds.BitsAllocated = 16; ds.BitsStored = 16; ds.HighBit = 15; ds.PixelRepresentation = 0
        ds.PixelData = np.ascontiguousarray(a[:, :, k].T).tobytes()
        if ts != ExplicitVRLittleEndian and not plain:
            ds.compress(ts, encoding_plugin='pylibjpeg' if ts == JPEG2000Lossless else None) if ts == JPEG2000Lossless else ds.compress(ts)
        ds.save_as(os.path.join(folder, 'IM%04d' % (k + 1)), enforce_file_format=True)


for i, (f, desc, tr, te, ti, c, ts) in enumerate(SEQ):
    im = nib.load(os.path.join(SRC, f + '.nii.gz'))
    write_series(np.asarray(im.dataobj, dtype=np.float32), im.affine, desc, i + 2, tr, te, ti, c, ts, os.path.join(OUT, 'S%02d' % (i + 2)))
# localiser: three slices of the T1, 2 mm spacing
im = nib.load(os.path.join(SRC, 't1.nii.gz'))
loc = np.asarray(im.dataobj, dtype=np.float32)[:, :, 10:13]
write_series(loc, im.affine, 'localizer', 1, 8, 4, None, False, ExplicitVRLittleEndian, os.path.join(OUT, 'S01'))
print('written to', OUT)

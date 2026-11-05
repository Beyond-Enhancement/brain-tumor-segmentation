#!/usr/bin/env python3
"""DICOM variants not covered by make_dicom.py: T1 as JPEG lossless (process 14, encoded with
imagecodecs/liblj92) and T1c as one enhanced multi-frame MR file with its frames stored in
reverse order. Usage: make_dicom_variants.py <demo dir> <out dir>"""
import os, sys, numpy as np, nibabel as nib, pydicom, imagecodecs
from pydicom.dataset import Dataset, FileMetaDataset
from pydicom.sequence import Sequence
from pydicom.encaps import encapsulate
from pydicom.uid import generate_uid, ExplicitVRLittleEndian, JPEGLossless, MRImageStorage, EnhancedMRImageStorage
SRC, OUT = sys.argv[1], sys.argv[2]
study = generate_uid()

def geom(im):
    A = im.affine.copy(); A[:2, :] *= -1
    cdir = A[:3, 0] / np.linalg.norm(A[:3, 0]); rdir = A[:3, 1] / np.linalg.norm(A[:3, 1])
    return A, cdir, rdir, np.linalg.norm(A[:3, 0]), np.linalg.norm(A[:3, 1]), np.linalg.norm(A[:3, 2])

def base(sop, ts, desc, num, series):
    ds = Dataset(); ds.file_meta = FileMetaDataset(); ds.file_meta.MediaStorageSOPClassUID = sop
    ds.file_meta.MediaStorageSOPInstanceUID = generate_uid(); ds.file_meta.TransferSyntaxUID = ts
    ds.SOPClassUID = sop; ds.SOPInstanceUID = ds.file_meta.MediaStorageSOPInstanceUID
    ds.PatientName = 'CANARY^ZEBRA'; ds.PatientID = 'CANARY-7741'; ds.StudyInstanceUID = study; ds.SeriesInstanceUID = series
    ds.Modality = 'MR'; ds.SeriesDescription = desc; ds.SeriesNumber = num
    ds.SamplesPerPixel = 1; ds.PhotometricInterpretation = 'MONOCHROME2'; ds.BitsAllocated = 16; ds.BitsStored = 16; ds.HighBit = 15; ds.PixelRepresentation = 0
    return ds

# T1: JPEG lossless, one file per slice
im = nib.load(os.path.join(SRC, 't1.nii.gz')); a = np.clip(np.round(np.asarray(im.dataobj, np.float32)), 0, 65535).astype(np.uint16)
A, cdir, rdir, dx, dy, dz = geom(im); series = generate_uid(); d = os.path.join(OUT, 'S02'); os.makedirs(d, exist_ok=True)
for k in range(a.shape[2]):
    ds = base(MRImageStorage, JPEGLossless, 't1_se_tra', 2, series)
    ds.InstanceNumber = k + 1; ds.RepetitionTime = 550; ds.EchoTime = 11
    ds.ImagePositionPatient = [float(v) for v in A[:3, :3] @ [0, 0, k] + A[:3, 3]]
    ds.ImageOrientationPatient = [float(v) for v in list(cdir) + list(rdir)]
    ds.PixelSpacing = [float(dy), float(dx)]; ds.SliceThickness = float(dz)
    ds.Rows, ds.Columns = a.shape[1], a.shape[0]
    frame = np.ascontiguousarray(a[:, :, k].T)
    enc = imagecodecs.ljpeg_encode(frame, bitspersample=16)
    assert np.array_equal(imagecodecs.ljpeg_decode(enc), frame)
    ds.PixelData = encapsulate([enc]); ds['PixelData'].VR = 'OB'
    ds.save_as(os.path.join(d, 'IM%04d' % (k + 1)), enforce_file_format=True)

# T1c: enhanced multi-frame, frames in reverse slice order
im = nib.load(os.path.join(SRC, 't1c.nii.gz')); a = np.clip(np.round(np.asarray(im.dataobj, np.float32)), 0, 65535).astype(np.uint16)
A, cdir, rdir, dx, dy, dz = geom(im); d = os.path.join(OUT, 'S03'); os.makedirs(d, exist_ok=True)
ds = base(EnhancedMRImageStorage, ExplicitVRLittleEndian, 't1_se_tra_post_gd', 3, generate_uid())
ds.ContrastBolusAgent = 'GADOVIST'; ds.InstanceNumber = 1
n = a.shape[2]; order = list(range(n))[::-1]
ds.NumberOfFrames = n; ds.Rows, ds.Columns = a.shape[1], a.shape[0]
sh = Dataset()
pm = Dataset(); pm.PixelSpacing = [float(dy), float(dx)]; pm.SliceThickness = float(dz)
po = Dataset(); po.ImageOrientationPatient = [float(v) for v in list(cdir) + list(rdir)]
sh.PixelMeasuresSequence = Sequence([pm]); sh.PlaneOrientationSequence = Sequence([po])
ds.SharedFunctionalGroupsSequence = Sequence([sh])
pf = []
for k in order:
    fg = Dataset(); pp = Dataset(); pp.ImagePositionPatient = [float(v) for v in A[:3, :3] @ [0, 0, k] + A[:3, 3]]
    fg.PlanePositionSequence = Sequence([pp]); pf.append(fg)
ds.PerFrameFunctionalGroupsSequence = Sequence(pf)
ds.PixelData = b''.join(np.ascontiguousarray(a[:, :, k].T).tobytes() for k in order)
ds.save_as(os.path.join(d, 'MF0001'), enforce_file_format=True)
print('written', OUT)

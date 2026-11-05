// Sequence assignment from NIfTI file names, including assignment by elimination.
import * as DCM from '../../app/js/lib/dicom.js';
const cases = [
  ['17_AX_T1_PRE.nii.gz', '20_SAG_MPR.nii.gz', '15_AX_T2.nii.gz', '8_SAG_3D_FLAIR.nii.gz'],
  ['t1.nii.gz', 't1c.nii.gz', 't2.nii.gz', 'flair.nii.gz'],
  ['BraTS-GLI-00001-000-t1n.nii.gz', 'BraTS-GLI-00001-000-t1c.nii.gz', 'BraTS-GLI-00001-000-t2w.nii.gz', 'BraTS-GLI-00001-000-t2f.nii.gz'],
  ['case_0000.nii.gz', 'case_0001.nii.gz', 'case_0002.nii.gz', 'case_0003.nii.gz'],
  ['a.nii.gz', 'b.nii.gz', 't2.nii.gz', 'flair.nii.gz'],            // two unknown: no elimination
  ['T1.nii.gz', 'T1c.nii.gz', 'T2.nii.gz', 'FLAIR.nii.gz', 'dwi.nii.gz'], // all known, extra file ignored
];
for (const names of cases) {
  const items = names.map(n => ({ id: n, scores: DCM.nameRoleScores(n) }));
  const g = DCM.assignRoles(items);
  const inf = DCM.fillByElimination(g, names);
  console.log(names.join(' | '), '->', JSON.stringify(g), inf ? '(by elimination: ' + inf + ')' : '');
}

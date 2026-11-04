import SimpleITK as sitk, numpy as np, sys, json
D=sys.argv[1]; fixed=sys.argv[2]; movs=sys.argv[3:]
f=sitk.Cast(sitk.ReadImage(D+fixed),sitk.sitkFloat32)
res={}
for mn in movs:
    m=sitk.Cast(sitk.ReadImage(D+mn),sitk.sitkFloat32)
    best=None
    for init in ['identity','moments']:
        if init=='identity':
            tx=sitk.Euler3DTransform(); tx.SetCenter(f.TransformContinuousIndexToPhysicalPoint([s/2 for s in f.GetSize()]))
        else: tx=sitk.CenteredTransformInitializer(f,m,sitk.Euler3DTransform(),sitk.CenteredTransformInitializerFilter.MOMENTS)
        R=sitk.ImageRegistrationMethod(); R.SetMetricAsMattesMutualInformation(48); R.SetMetricSamplingStrategy(R.RANDOM); R.SetMetricSamplingPercentage(0.2,1)
        R.SetInterpolator(sitk.sitkLinear); R.SetOptimizerAsRegularStepGradientDescent(2.0,1e-4,300,0.5); R.SetOptimizerScalesFromPhysicalShift()
        R.SetShrinkFactorsPerLevel([4,2,1]); R.SetSmoothingSigmasPerLevel([2,1,0]); R.SmoothingSigmasAreSpecifiedInPhysicalUnitsOn()
        R.SetInitialTransform(tx,inPlace=False)
        out=R.Execute(f,m); v=R.GetMetricValue()
        e=out
        while e.GetName()=='CompositeTransform': e=e.GetNthTransform(0)
        e=sitk.Euler3DTransform(e)
        print(mn,init,'metric %.4f'%v,'angles(deg)',np.round(np.degrees([e.GetAngleX(),e.GetAngleY(),e.GetAngleZ()]),2),'t(LPS)',np.round(e.GetTranslation(),2))
        if best is None or v<best[0]: best=(v,e)
    # 4x4 fixed-world(RAS)->moving-world(RAS) matrix
    M=np.array(best[1].GetMatrix()).reshape(3,3); c=np.array(best[1].GetCenter()); t=np.array(best[1].GetTranslation())
    off=t+c-M@c
    F=np.diag([-1,-1,1.]); A=np.eye(4); A[:3,:3]=F@M@F; A[:3,3]=F@off
    res[mn]=A.tolist()
json.dump(res,open('/tmp/claude-0/-home-claude/6206506b-0c74-5d1e-bd38-d9a96d8d4346/scratchpad/sitk_T.json','w'))

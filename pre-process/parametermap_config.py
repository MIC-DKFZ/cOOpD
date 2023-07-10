import SimpleITK as sitk


def parameter_file():

    parameterMapVector_affine_bspline1 = sitk.VectorOfParameterMap()
    parameterMapVector_affine_bspline1_bspline2 = sitk.VectorOfParameterMap()


    affineparammap = sitk.ParameterMap()
    affineparammap['Registration'] = ['MultiResolutionRegistration']
    affineparammap['FixedImagePyramid'] = ['FixedRecursiveImagePyramid']
    affineparammap['MovingImagePyramid'] = ['MovingRecursiveImagePyramid']
    affineparammap['Interpolator'] = ['BSplineInterpolator']
    affineparammap['Metric'] = ['AdvancedNormalizedCorrelation']
    affineparammap['Optimizer'] = ['AdaptiveStochasticGradientDescent']
    affineparammap['ResampleInterpolator'] = ['FinalBSplineInterpolator']
    affineparammap['Resampler'] = ['DefaultResampler']
    affineparammap['Transform'] = ['AffineTransform']

    affineparammap['NumberOfResolutions'] = ['5']
    affineparammap['ImagePyramidSchedule'] = ['16', '16', '16', '8', '8', '8', '4', '4', '4', '2', '2', '2', '1', '1', '1']
    affineparammap['AutomaticScalesEstimation'] = ['true']
    affineparammap['AutomaticTransformInitialization'] = ['true']
    affineparammap['HowToCombineTransforms'] = ['Compose']
    affineparammap['AutomaticParameterEstimation'] = ['true']
    affineparammap['UseAdaptiveStepSizes'] = ['true']

    affineparammap['WriteTransformParametersEachIteration'] = ['false']
    affineparammap['WriteTransformParametersEachResolution'] = ['true']
    affineparammap['WriteResultImageAfterEachResolution'] = ['true']
    affineparammap['ShowExactMetricValue'] = ['false']
    affineparammap['ErodeMask'] = ['false']
    affineparammap['UseDirectionCosines'] = ['true']


    affineparammap['ImageSampler'] = ['RandomCoordinate']
    affineparammap['NumberOfSpatialSamples'] = ['2000']
    affineparammap['NewSamplesEveryIteration'] = ['true']
    affineparammap['UseRandomSampleRegion'] = ['false']
    affineparammap['MaximumNumberOfSamplingAttempts'] = ['5']

    affineparammap['BSplineInterpolationOrder'] = ['1']
    affineparammap['FinalBSplineInterpolationOrder'] = ['3']
    affineparammap['DefaultPixelValue'] = ['0']

    parameterMapVector_affine_bspline1.append(affineparammap)
    parameterMapVector_affine_bspline1_bspline2.append(affineparammap)

    # 1st b-spline

    bsplineparammap_1 = sitk.ParameterMap()
    bsplineparammap_1['Registration'] = ['MultiMetricMultiResolutionRegistration']
    bsplineparammap_1['FixedImagePyramid'] = ['FixedRecursiveImagePyramid']
    bsplineparammap_1['MovingImagePyramid'] = ['MovingRecursiveImagePyramid']
    bsplineparammap_1['Interpolator'] = ['BSplineInterpolator']
    bsplineparammap_1['Metric'] = ['AdvancedNormalizedCorrelation', 'TransformBendingEnergyPenalty']
    bsplineparammap_1['Optimizer'] = ['AdaptiveStochasticGradientDescent']
    bsplineparammap_1['ResampleInterpolator'] = ['FinalBSplineInterpolator']
    bsplineparammap_1['Resampler'] = ['DefaultResampler']
    bsplineparammap_1['Transform'] = ['BSplineTransform']

    bsplineparammap_1['NumberOfResolutions'] = ['5']
    bsplineparammap_1['ImagePyramidSchedule'] = ['16', '16', '16', '8', '8', '8', '4', '4', '4', '2', '2', '2', '1', '1', '1']

    bsplineparammap_1['FinalGridSpacingInPhysicalUnits'] = ['10', '10', '10']
    bsplineparammap_1['GridSpacingSchedule'] = ['8', '8', '4', '2', '1']
    bsplineparammap_1['HowToCombineTransforms'] = ['Compose']

    bsplineparammap_1['MaximumNumberOfIterations'] = ['1000']
    bsplineparammap_1['UseAdaptiveStepSizes'] = ['true']

    bsplineparammap_1['UseRelativeWeights'] = ['true']
    bsplineparammap_1['Metric0RelativeWeight'] = ['1']
    bsplineparammap_1['Metric1RelativeWeight'] = ['0.05']

    bsplineparammap_1['WriteTransformParametersEachIteration'] = ['false']
    bsplineparammap_1['WriteTransformParametersEachResolution'] = ['true']
    bsplineparammap_1['WriteResultImageAfterEachResolution'] = ['false']
    bsplineparammap_1['WritePyramidImagesAfterEachResolution'] = ['false']
    bsplineparammap_1['ShowExactMetricValue'] = ['false']
    bsplineparammap_1['ErodeMask'] = ['false']
    bsplineparammap_1['UseDirectionCosines'] = ['true']

    bsplineparammap_1['ImageSampler'] = ['RandomCoordinate']
    bsplineparammap_1['NumberOfSpatialSamples'] = ['2000']
    bsplineparammap_1['NewSamplesEveryIteration'] = ['true']
    bsplineparammap_1['SampleRegionSize'] = ['50', '50', '50']
    bsplineparammap_1['MaximumNumberOfSamplingAttempts'] = ['50']

    bsplineparammap_1['BSplineInterpolationOrder'] = ['1']
    bsplineparammap_1['FinalBSplineInterpolationOrder'] = ['3']
    bsplineparammap_1['DefaultPixelValue'] = ['0']

    parameterMapVector_affine_bspline1.append(bsplineparammap_1)
    parameterMapVector_affine_bspline1_bspline2.append(bsplineparammap_1)
    # 2st b-spline

    bsplineparammap_2 = sitk.ParameterMap()
    bsplineparammap_2['Registration'] = ['MultiMetricMultiResolutionRegistration']
    bsplineparammap_2['FixedImagePyramid'] = ['FixedRecursiveImagePyramid']
    bsplineparammap_2['MovingImagePyramid'] = ['MovingRecursiveImagePyramid']
    bsplineparammap_2['Interpolator'] = ['BSplineInterpolator']
    bsplineparammap_2['Metric'] = ['AdvancedNormalizedCorrelation', 'TransformBendingEnergyPenalty']
    bsplineparammap_2['Optimizer'] = ['AdaptiveStochasticGradientDescent']
    bsplineparammap_2['ResampleInterpolator'] = ['FinalBSplineInterpolator']
    bsplineparammap_2['Resampler'] = ['DefaultResampler']
    bsplineparammap_2['Transform'] = ['BSplineTransform']

    bsplineparammap_2['NumberOfResolutions'] = ['5']
    bsplineparammap_2['ImagePyramidSchedule'] = ['4', '4', '4', '3', '3', '3', '2', '2', '2', '1', '1', '1', '1', '1', '1']

    bsplineparammap_2['FinalGridSpacingInPhysicalUnits'] = ['5', '5', '5']
    bsplineparammap_2['GridSpacingSchedule'] = ['16', '8', '4', '2', '1']
    bsplineparammap_2['HowToCombineTransforms'] = ['Compose']

    bsplineparammap_2['MaximumNumberOfIterations'] = ['2000']
    bsplineparammap_2['UseAdaptiveStepSizes'] = ['true']

    bsplineparammap_2['UseRelativeWeights'] = ['true']
    bsplineparammap_2['Metric0RelativeWeight'] = ['1']
    bsplineparammap_2['Metric1RelativeWeight'] = ['0.05']

    bsplineparammap_2['WriteTransformParametersEachIteration'] = ['false']
    bsplineparammap_2['WriteTransformParametersEachResolution'] = ['true']
    bsplineparammap_2['WriteResultImageAfterEachResolution'] = ['false']
    bsplineparammap_2['WritePyramidImagesAfterEachResolution'] = ['false']
    bsplineparammap_2['ShowExactMetricValue'] = ['false']
    bsplineparammap_2['ErodeMask'] = ['false', 'false', 'true', 'true', 'true']
    bsplineparammap_2['UseDirectionCosines'] = ['true']

    bsplineparammap_2['ImageSampler'] = ['RandomCoordinate']
    bsplineparammap_2['NumberOfSpatialSamples'] = ['2000']
    bsplineparammap_2['NewSamplesEveryIteration'] = ['true']
    bsplineparammap_2['UseRandomSampleRegion'] = ['false']
    bsplineparammap_2['SampleRegionSize'] = ['50', '50', '50']
    bsplineparammap_2['MaximumNumberOfSamplingAttempts'] = ['50']

    bsplineparammap_2['BSplineInterpolationOrder'] = ['1']
    bsplineparammap_2['FinalBSplineInterpolationOrder'] = ['3']
    bsplineparammap_2['DefaultPixelValue'] = ['0']

    parameterMapVector_affine_bspline1_bspline2.append(bsplineparammap_2)

    return affineparammap, bsplineparammap_1, bsplineparammap_2, parameterMapVector_affine_bspline1, parameterMapVector_affine_bspline1_bspline2


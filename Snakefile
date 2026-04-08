rule draw_pe_samples:
    input:
        "src/data/pe-samples-raw/IGWN-GWTC2p1-v2-GW190408_181802_PEDataRelease_mixed_nocosmo.h5",
        "src/data/pe-samples-raw/IGWN-GWTC2p1-v2-GW190412_053044_PEDataRelease_mixed_nocosmo.h5",
        "src/data/pe-samples-raw/IGWN-GWTC2p1-v2-GW190413_052954_PEDataRelease_mixed_nocosmo.h5",
        "src/data/pe-samples-raw/IGWN-GWTC2p1-v2-GW190413_134308_PEDataRelease_mixed_nocosmo.h5",
        "src/data/pe-samples-raw/IGWN-GWTC2p1-v2-GW190421_213856_PEDataRelease_mixed_nocosmo.h5",
        "src/data/pe-samples-raw/IGWN-GWTC2p1-v2-GW190503_185404_PEDataRelease_mixed_nocosmo.h5",
        "src/data/pe-samples-raw/IGWN-GWTC2p1-v2-GW190512_180714_PEDataRelease_mixed_nocosmo.h5",
        "src/data/pe-samples-raw/IGWN-GWTC2p1-v2-GW190513_205428_PEDataRelease_mixed_nocosmo.h5",
        "src/data/pe-samples-raw/IGWN-GWTC2p1-v2-GW190517_055101_PEDataRelease_mixed_nocosmo.h5",
        "src/data/pe-samples-raw/IGWN-GWTC2p1-v2-GW190519_153544_PEDataRelease_mixed_nocosmo.h5",
        "src/data/pe-samples-raw/IGWN-GWTC2p1-v2-GW190521_030229_PEDataRelease_mixed_nocosmo.h5",
        "src/data/pe-samples-raw/IGWN-GWTC2p1-v2-GW190521_074359_PEDataRelease_mixed_nocosmo.h5",
        "src/data/pe-samples-raw/IGWN-GWTC2p1-v2-GW190527_092055_PEDataRelease_mixed_nocosmo.h5",
        "src/data/pe-samples-raw/IGWN-GWTC2p1-v2-GW190602_175927_PEDataRelease_mixed_nocosmo.h5",
        "src/data/pe-samples-raw/IGWN-GWTC2p1-v2-GW190620_030421_PEDataRelease_mixed_nocosmo.h5",
        "src/data/pe-samples-raw/IGWN-GWTC2p1-v2-GW190630_185205_PEDataRelease_mixed_nocosmo.h5",
        "src/data/pe-samples-raw/IGWN-GWTC2p1-v2-GW190701_203306_PEDataRelease_mixed_nocosmo.h5",
        "src/data/pe-samples-raw/IGWN-GWTC2p1-v2-GW190706_222641_PEDataRelease_mixed_nocosmo.h5",
        "src/data/pe-samples-raw/IGWN-GWTC2p1-v2-GW190707_093326_PEDataRelease_mixed_nocosmo.h5",
        "src/data/pe-samples-raw/IGWN-GWTC2p1-v2-GW190708_232457_PEDataRelease_mixed_nocosmo.h5",
        "src/data/pe-samples-raw/IGWN-GWTC2p1-v2-GW190719_215514_PEDataRelease_mixed_nocosmo.h5",
        "src/data/pe-samples-raw/IGWN-GWTC2p1-v2-GW190720_000836_PEDataRelease_mixed_nocosmo.h5",
        "src/data/pe-samples-raw/IGWN-GWTC2p1-v2-GW190727_060333_PEDataRelease_mixed_nocosmo.h5",
        "src/data/pe-samples-raw/IGWN-GWTC2p1-v2-GW190728_064510_PEDataRelease_mixed_nocosmo.h5",
        "src/data/pe-samples-raw/IGWN-GWTC2p1-v2-GW190731_140936_PEDataRelease_mixed_nocosmo.h5",
        "src/data/pe-samples-raw/IGWN-GWTC2p1-v2-GW190803_022701_PEDataRelease_mixed_nocosmo.h5",
        "src/data/pe-samples-raw/IGWN-GWTC2p1-v2-GW190828_063405_PEDataRelease_mixed_nocosmo.h5",
        "src/data/pe-samples-raw/IGWN-GWTC2p1-v2-GW190828_065509_PEDataRelease_mixed_nocosmo.h5",
        "src/data/pe-samples-raw/IGWN-GWTC2p1-v2-GW190910_112807_PEDataRelease_mixed_nocosmo.h5",
        "src/data/pe-samples-raw/IGWN-GWTC2p1-v2-GW190915_235702_PEDataRelease_mixed_nocosmo.h5",
        "src/data/pe-samples-raw/IGWN-GWTC2p1-v2-GW190924_021846_PEDataRelease_mixed_nocosmo.h5",
        "src/data/pe-samples-raw/IGWN-GWTC2p1-v2-GW190929_012149_PEDataRelease_mixed_nocosmo.h5",
        "src/data/pe-samples-raw/IGWN-GWTC2p1-v2-GW190930_133541_PEDataRelease_mixed_nocosmo.h5",
        "src/data/pe-samples-raw/IGWN-GWTC3p0-v1-GW191103_012549_PEDataRelease_mixed_nocosmo.h5",
        "src/data/pe-samples-raw/IGWN-GWTC3p0-v1-GW191105_143521_PEDataRelease_mixed_nocosmo.h5",
        "src/data/pe-samples-raw/IGWN-GWTC3p0-v1-GW191109_010717_PEDataRelease_mixed_nocosmo.h5",
        "src/data/pe-samples-raw/IGWN-GWTC3p0-v1-GW191127_050227_PEDataRelease_mixed_nocosmo.h5",
        "src/data/pe-samples-raw/IGWN-GWTC3p0-v1-GW191129_134029_PEDataRelease_mixed_nocosmo.h5",
        "src/data/pe-samples-raw/IGWN-GWTC3p0-v1-GW191204_171526_PEDataRelease_mixed_nocosmo.h5",
        "src/data/pe-samples-raw/IGWN-GWTC3p0-v1-GW191215_223052_PEDataRelease_mixed_nocosmo.h5",
        "src/data/pe-samples-raw/IGWN-GWTC3p0-v1-GW191216_213338_PEDataRelease_mixed_nocosmo.h5",
        "src/data/pe-samples-raw/IGWN-GWTC3p0-v1-GW191222_033537_PEDataRelease_mixed_nocosmo.h5",
        "src/data/pe-samples-raw/IGWN-GWTC3p0-v1-GW191230_180458_PEDataRelease_mixed_nocosmo.h5",
        "src/data/pe-samples-raw/IGWN-GWTC3p0-v1-GW200112_155838_PEDataRelease_mixed_nocosmo.h5",
        "src/data/pe-samples-raw/IGWN-GWTC3p0-v1-GW200128_022011_PEDataRelease_mixed_nocosmo.h5",
        "src/data/pe-samples-raw/IGWN-GWTC3p0-v1-GW200129_065458_PEDataRelease_mixed_nocosmo.h5",
        "src/data/pe-samples-raw/IGWN-GWTC3p0-v1-GW200202_154313_PEDataRelease_mixed_nocosmo.h5",
        "src/data/pe-samples-raw/IGWN-GWTC3p0-v1-GW200208_130117_PEDataRelease_mixed_nocosmo.h5",
        "src/data/pe-samples-raw/IGWN-GWTC3p0-v1-GW200209_085452_PEDataRelease_mixed_nocosmo.h5",
        "src/data/pe-samples-raw/IGWN-GWTC3p0-v1-GW200216_220804_PEDataRelease_mixed_nocosmo.h5",
        "src/data/pe-samples-raw/IGWN-GWTC3p0-v1-GW200219_094415_PEDataRelease_mixed_nocosmo.h5",
        "src/data/pe-samples-raw/IGWN-GWTC3p0-v1-GW200224_222234_PEDataRelease_mixed_nocosmo.h5",
        "src/data/pe-samples-raw/IGWN-GWTC3p0-v1-GW200225_060421_PEDataRelease_mixed_nocosmo.h5",
        "src/data/pe-samples-raw/IGWN-GWTC3p0-v1-GW200302_015811_PEDataRelease_mixed_nocosmo.h5",
        "src/data/pe-samples-raw/IGWN-GWTC3p0-v1-GW200311_115853_PEDataRelease_mixed_nocosmo.h5",
        "src/data/pe-samples-raw/IGWN-GWTC3p0-v1-GW200316_215756_PEDataRelease_mixed_nocosmo.h5"
    output:
        "src/data/pe-samples.h5"
    cache:
        True
    script:
        "src/scripts/draw_pe_samples.py"

rule draw_selection_samples:
    input:
        "src/data/endo3_bbhpop-LIGO-T2100113-v12.hdf5"
    output:
        "src/data/selection-samples.h5"
    cache:
        True
    script:
        "src/scripts/draw_selection_samples.py"

rule sample:
    input:
        "src/data/pe-samples.h5",
        "src/data/selection-samples.h5"
    output:
        "src/data/trace.nc"
    cache:
        True
    script:
        "src/scripts/run_fit.py"

rule sample_cosmo:
    input:
        "src/data/pe-samples.h5",
        "src/data/selection-samples.h5"
    output:
        "src/data/trace_cosmo.nc"
    cache:
        True
    script:
        "src/scripts/run_cosmo_fit.py"

rule mock_injections_truth:
    output:
        "src/data/mock_injections.h5"
    cache:
        True
    script:
        "src/scripts/mock_injections.py"

rule mock_observations:
    input:
        "src/data/mock_injections.h5"
    output:
        "src/data/mock_observations.h5"
    cache:
        True
    script:
        "src/scripts/mock_observations.py"

rule mock_year_samples:
    input:
        "src/data/mock_injections.h5",
        "src/data/mock_observations.h5"
    output:
        "src/data/mock_year_samples.h5"
    cache:
        True
    script:
        "src/scripts/mock_one_year_samples.py"
    

from datasets import DatasetMRI

dataset_dir = "/shared/workspace/lui/AUTH_PERSONEL_ONLY_diferre/adni/pre_mri"
glob_regx = "*/*/*/*.BET.FLIRT.nii.gz"
dataset = DatasetMRI(dataset_dir=dataset_dir, h5dataset_path=dataset_dir, h5dataset_name="adni_all", glob_regx=glob_regx)

dataset.set_data_pointer() 
